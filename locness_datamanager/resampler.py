"""
Persistent resampler for raw sensor data.

This module provides a stateful resampler that:
1. Tracks which raw data has been processed to avoid duplicates
2. Maintains pH history for moving average calculations
3. Efficiently handles incremental resampling operations
"""

import sqlite3
import pandas as pd
import logging
from typing import Optional, Dict
from isfetphcalc import calc_ph


class PersistentResampler:
    """
    A stateful resampler that maintains processed data tracking and pH history.
    
    This class ensures that each raw data row is only used once in resampling
    and efficiently maintains the necessary pH history for moving average calculations.
    """
    
    def __init__(self, 
                 sqlite_path: str,
                 resample_interval: str = '10s',
                 ph_ma_window: int = 120,
                 ph_freq: float = 0.5,
                 ph_k0: float = 0.0,
                 ph_k2: float = 0.0):
        """
        Initialize the persistent resampler.
        
        Args:
            sqlite_path: Path to SQLite database
            resample_interval: Resampling interval (e.g., '10s')
            ph_ma_window: pH moving average window in seconds
            ph_freq: pH sampling frequency in Hz
            ph_k0: pH calibration coefficient k0
            ph_k2: pH calibration coefficient k2
        """
        self.sqlite_path = sqlite_path
        self.resample_interval = resample_interval
        self.ph_ma_window = ph_ma_window
        self.ph_freq = ph_freq
        self.ph_k0 = ph_k0
        self.ph_k2 = ph_k2
        
        # Track processed raw data by table and max datetime_utc
        self.last_processed: Dict[str, Optional[pd.Timestamp]] = {
            'rhodamine': None,
            'ph': None,
            'tsg': None,
            'gps': None
        }
        
        # Buffer for pH data needed for moving averages
        self.ph_buffer = pd.DataFrame()
        
        # Buffer size (number of samples to keep for moving average)
        self.ph_buffer_size = max(200, int(ph_ma_window * ph_freq * 2))  # Keep 2x window size
        
        logging.info(f"Initialized PersistentResampler with interval={resample_interval}, "
                    f"pH window={ph_ma_window}s, freq={ph_freq}Hz")
    
    def get_new_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get new raw data that hasn't been processed yet.
        
        Returns:
            Dict mapping table names to DataFrames of new data
        """
        new_data = {}
        
        try:
            conn = sqlite3.connect(self.sqlite_path)
            
            # Define table columns
            table_columns = {
                'rhodamine': ['datetime_utc', 'rho_ppb'],
                'ph': ['datetime_utc', 'vrse', 'ph_total'],
                'tsg': ['datetime_utc', 'temp', 'salinity'],
                'gps': ['datetime_utc', 'latitude', 'longitude']
            }
            
            for table, columns in table_columns.items():
                # Build query with timestamp filter
                base_query = f"SELECT {', '.join(columns)} FROM {table}"
                
                if self.last_processed[table] is not None:
                    # Get only records after the last processed timestamp
                    unix_timestamp = int(self.last_processed[table].timestamp())
                    query = f"{base_query} WHERE datetime_utc > {unix_timestamp} ORDER BY datetime_utc"
                else:
                    # First run, get all data
                    query = f"{base_query} ORDER BY datetime_utc"
                
                try:
                    df = pd.read_sql_query(query, conn)
                    
                    if not df.empty:
                        # Convert timestamps
                        if df['datetime_utc'].dtype in ['int64', 'int32']:
                            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='s')
                        else:
                            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
                        
                        # Update last processed timestamp
                        self.last_processed[table] = df['datetime_utc'].max()
                        
                        logging.info(f"Found {len(df)} new records in {table}, "
                                   f"latest: {self.last_processed[table]}")
                    
                    new_data[table] = df
                    
                except Exception as e:
                    logging.error(f"Error reading table {table}: {e}")
                    new_data[table] = pd.DataFrame(columns=columns)
            
            conn.close()
            
        except Exception as e:
            logging.error(f"Error connecting to database: {e}")
            # Return empty DataFrames
            for table in ['rhodamine', 'ph', 'tsg', 'gps']:
                new_data[table] = pd.DataFrame()
        
        return new_data
    
    def resample_and_join(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Resample raw data tables and join into a single DataFrame.
        
        Args:
            raw_data: Dict mapping table names to DataFrames
            
        Returns:
            Resampled and joined DataFrame
        """
        expected_cols = ['datetime_utc', 'latitude', 'longitude', 'rho_ppb', 
                        'ph_total', 'vrse', 'temp', 'salinity']
        
        # Check if all DataFrames are empty
        if all(df.empty for df in raw_data.values()):
            return pd.DataFrame(columns=expected_cols)
        
        resampled_dfs = []
        
        for table, df in raw_data.items():
            if df.empty or 'datetime_utc' not in df.columns:
                continue
                
            # Prepare DataFrame for resampling
            df_prep = df.copy()
            df_prep = df_prep.drop_duplicates(subset='datetime_utc')
            df_prep = df_prep.set_index('datetime_utc')
            
            # Resample to common interval using nearest
            df_resampled = df_prep.resample(self.resample_interval).nearest()
            
            if not df_resampled.empty:
                resampled_dfs.append(df_resampled)
        
        if not resampled_dfs:
            return pd.DataFrame(columns=expected_cols)
        
        # Join all resampled data on datetime_utc
        result = resampled_dfs[0]
        for df in resampled_dfs[1:]:
            result = result.join(df, how='outer')
        
        result = result.reset_index()
        
        # Ensure all expected columns exist
        for col in expected_cols:
            if col not in result.columns:
                result[col] = pd.NA
        
        return result[expected_cols]
    
    def add_corrected_ph(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add corrected pH column to DataFrame."""
        if 'temp' not in df.columns or 'salinity' not in df.columns or 'vrse' not in df.columns:
            df['ph_corrected'] = pd.NA
            return df
        
        # Calculate corrected pH
        mask = df['temp'].isna() | df['salinity'].isna() | df['vrse'].isna()
        
        try:
            ph_free, ph_total = calc_ph(
                Vrs=df['vrse'],
                Press=0,
                Temp=df['temp'],
                Salt=df['salinity'],
                k0=self.ph_k0,
                k2=self.ph_k2,
                Pcoefs=0
            )
            df['ph_corrected'] = ph_total
        except Exception as e:
            logging.warning(f"Error calculating corrected pH: {e}")
            df['ph_corrected'] = pd.NA
        
        # Set NaN for rows with missing input data
        if mask.any():
            df.loc[mask, 'ph_corrected'] = pd.NA
        
        return df
    
    def update_ph_buffer(self, new_data: pd.DataFrame):
        """
        Update the pH buffer with new data for moving average calculations.
        
        Args:
            new_data: New resampled data containing pH information
        """
        if new_data.empty:
            return
        
        # Extract pH-relevant columns
        ph_cols = ['datetime_utc', 'ph_total', 'ph_corrected']
        new_ph_data = new_data[ph_cols].dropna(subset=['ph_total'])
        
        if new_ph_data.empty:
            return
        
        # Add to buffer
        if self.ph_buffer.empty:
            self.ph_buffer = new_ph_data.copy()
        else:
            self.ph_buffer = pd.concat([self.ph_buffer, new_ph_data], ignore_index=True)
        
        # Sort by datetime and keep only recent data
        self.ph_buffer = self.ph_buffer.sort_values('datetime_utc')
        
        # Trim buffer to keep only necessary data
        if len(self.ph_buffer) > self.ph_buffer_size:
            self.ph_buffer = self.ph_buffer.tail(self.ph_buffer_size).reset_index(drop=True)
        
        logging.debug(f"pH buffer updated, now contains {len(self.ph_buffer)} records")
    
    def add_ph_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pH moving averages using the maintained pH buffer.
        
        Args:
            df: DataFrame with new resampled data
            
        Returns:
            DataFrame with moving average columns added
        """
        if df.empty:
            return df
        
        # Update pH buffer with new data
        self.update_ph_buffer(df)
        
        if self.ph_buffer.empty:
            # No pH data available
            df['ph_total_ma'] = pd.NA
            df['ph_corrected_ma'] = pd.NA
            return df
        
        # Calculate window size in samples
        window_size = max(1, int(self.ph_ma_window * self.ph_freq))
        
        # Calculate moving averages on the buffer
        buffer_with_ma = self.ph_buffer.copy()
        buffer_with_ma['ph_total_ma'] = buffer_with_ma['ph_total'].rolling(
            window=window_size, min_periods=1
        ).mean()
        buffer_with_ma['ph_corrected_ma'] = buffer_with_ma['ph_corrected'].rolling(
            window=window_size, min_periods=1
        ).mean()
        
        # Merge moving averages back to the result DataFrame
        df = df.merge(
            buffer_with_ma[['datetime_utc', 'ph_total_ma', 'ph_corrected_ma']], 
            on='datetime_utc', 
            how='left'
        )
        
        # Fill NaN moving averages for records without pH data
        if 'ph_total_ma' not in df.columns:
            df['ph_total_ma'] = pd.NA
        if 'ph_corrected_ma' not in df.columns:
            df['ph_corrected_ma'] = pd.NA
        
        return df
    
    def process_new_data(self) -> pd.DataFrame:
        """
        Process new raw data and return resampled results.
        
        This is the main method that:
        1. Gets new raw data (avoiding duplicates)
        2. Resamples and joins the data
        3. Adds corrected pH
        4. Adds moving averages
        
        Returns:
            DataFrame with processed new data
        """
        # Get new raw data
        raw_data = self.get_new_raw_data()
        
        # Check if any new data is available
        total_new_records = sum(len(df) for df in raw_data.values())
        if total_new_records == 0:
            logging.info("No new raw data to process")
            return pd.DataFrame()
        
        logging.info(f"Processing {total_new_records} new raw records")
        
        # Resample and join
        df = self.resample_and_join(raw_data)
        
        if df.empty:
            logging.info("No data after resampling")
            return df
        
        # Add corrected pH
        df = self.add_corrected_ph(df)
        
        # Add moving averages
        df = self.add_ph_moving_averages(df)
        
        logging.info(f"Generated {len(df)} resampled records")
        
        return df
    
    def reset_state(self):
        """Reset the resampler state (useful for testing or reprocessing all data)."""
        self.last_processed = {table: None for table in self.last_processed}
        self.ph_buffer = pd.DataFrame()
        logging.info("Resampler state reset")
    
    def get_state_summary(self) -> Dict:
        """Get a summary of the current resampler state."""
        return {
            'last_processed': {k: str(v) if v else None for k, v in self.last_processed.items()},
            'ph_buffer_size': len(self.ph_buffer),
            'ph_buffer_latest': str(self.ph_buffer['datetime_utc'].max()) if not self.ph_buffer.empty else None,
            'resample_interval': self.resample_interval,
            'ph_ma_window': self.ph_ma_window
        }
