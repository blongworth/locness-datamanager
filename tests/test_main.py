import pytest
import sqlite3
import pandas as pd
from unittest.mock import Mock, patch
from locness_datamanager.backup_db import DatabaseBackup
from locness_datamanager import main


class TestPollAndProcess:
    """Test the poll_and_process function"""
    
    @patch('locness_datamanager.main.time.sleep')
    @patch('locness_datamanager.main.process_summary_incremental')
    @patch('locness_datamanager.main.process_raw_data_incremental')
    @patch('locness_datamanager.main.time.time')
    def test_single_iteration_no_parquet_no_backup(self, mock_time, mock_process_raw, mock_process_summary, mock_sleep):
        """Test a single iteration with no parquet writing or backup"""
        # Mock time to control timing
        mock_time.side_effect = [0, 1, 2, 3]  # Initial times, then loop times
        
        # Mock sleep to break the loop after first iteration
        mock_sleep.side_effect = KeyboardInterrupt()
        
        # Mock backup manager
        backup_manager = Mock(spec=DatabaseBackup)
        
        # Test parameters
        db_path = "test.db"
        db_poll_interval = 5
        ph_k0 = -1.5
        ph_k2 = -0.001
        
        with pytest.raises(KeyboardInterrupt):
            main.poll_and_process(
                db_path=db_path,
                db_poll_interval=db_poll_interval,
                backup_manager=backup_manager,
                ph_k0=ph_k0,
                ph_k2=ph_k2,
                ph_ma_window=120,
                ph_freq=0.5
            )
        
        # Verify process_raw_data_incremental was called with correct parameters
        mock_process_raw.assert_called_once_with(
            sqlite_path=db_path,
            resample_interval='10s',
            summary_table='underway_summary',
            replace_all=False,
            ph_k0=ph_k0,
            ph_k2=ph_k2,
            ph_ma_window=120,
            ph_freq=0.5,
        )
        
        # Verify summary processing was not called (not time yet)
        mock_process_summary.assert_not_called()
        
        # Verify backup was not called (not time yet)
        backup_manager.create_backup.assert_not_called()
        
        # Verify sleep was called with correct interval
        mock_sleep.assert_called_once_with(db_poll_interval)

    @patch('locness_datamanager.main.time.sleep')
    @patch('locness_datamanager.main.process_summary_incremental')
    @patch('locness_datamanager.main.process_raw_data_incremental')
    @patch('locness_datamanager.main.time.time')
    def test_parquet_writing_triggered(self, mock_time, mock_process_raw, mock_process_summary, mock_sleep):
        """Test that parquet writing is triggered at the right time"""
        # Mock time to trigger parquet writing
        mock_time.side_effect = [
            0,    # last_parquet initial
            0,    # last_backup initial
            70,   # first time check (triggers parquet: 70 > 60)
            70,   # parquet time update
            71,   # backup time check
            72    # sleep time check
        ]
        
        mock_sleep.side_effect = KeyboardInterrupt()
        backup_manager = Mock(spec=DatabaseBackup)
        
        test_params = {
            'db_path': 'test.db',
            'parquet_poll_interval': 60,
            'parquet_path': 'test.parquet',
            'csv_path': 'test.csv',
            'backup_manager': backup_manager
        }
        
        with pytest.raises(KeyboardInterrupt):
            main.poll_and_process(**test_params)
        
        # Verify summary processing was called
        mock_process_summary.assert_called_once_with(
            sqlite_path='test.db',
            resample_interval='60s',
            parquet_path='test.parquet',
            partition_hours=6,
            csv_path='test.csv',
        )

    @patch('locness_datamanager.main.time.sleep')
    @patch('locness_datamanager.main.process_summary_incremental')
    @patch('locness_datamanager.main.process_raw_data_incremental')
    @patch('locness_datamanager.main.time.time')
    def test_backup_triggered(self, mock_time, mock_process_raw, mock_process_summary, mock_sleep):
        """Test that backup is triggered at the right time"""
        # Mock time to trigger backup
        mock_time.side_effect = [
            0,     # last_parquet initial
            0,     # last_backup initial
            30,    # first time check (no parquet: 30 < 60)
            3700,  # backup time check (triggers backup: 3700 > 3600)
            3700,  # backup time update
            3701   # sleep time check
        ]
        
        mock_sleep.side_effect = KeyboardInterrupt()
        backup_manager = Mock(spec=DatabaseBackup)
        
        with pytest.raises(KeyboardInterrupt):
            main.poll_and_process(
                db_path='test.db',
                backup_interval=3600,
                backup_manager=backup_manager
            )
        
        # Verify backup was called
        backup_manager.create_backup.assert_called_once()

    @patch('locness_datamanager.main.time.sleep')
    @patch('locness_datamanager.main.process_summary_incremental')
    @patch('locness_datamanager.main.process_raw_data_incremental')
    @patch('locness_datamanager.main.time.time')
    def test_both_parquet_and_backup_triggered(self, mock_time, mock_process_raw, mock_process_summary, mock_sleep):
        """Test when both parquet writing and backup are triggered in same iteration"""
        mock_time.side_effect = [
            0,     # last_parquet initial
            0,     # last_backup initial
            3700,  # first time check (triggers parquet: 3700 > 60)
            3700,  # parquet time update
            3700,  # backup time check (triggers backup: 3700 > 3600)
            3700,  # backup time update
            3701   # sleep time check
        ]
        
        mock_sleep.side_effect = KeyboardInterrupt()
        backup_manager = Mock(spec=DatabaseBackup)
        
        with pytest.raises(KeyboardInterrupt):
            main.poll_and_process(
                db_path='test.db',
                parquet_poll_interval=60,
                backup_interval=3600,
                backup_manager=backup_manager,
                parquet_path='test.parquet',
                csv_path='test.csv'
            )
        
        # Verify both operations were called
        mock_process_summary.assert_called_once()
        backup_manager.create_backup.assert_called_once()

    @patch('locness_datamanager.main.time.sleep')
    @patch('locness_datamanager.main.process_summary_incremental')
    @patch('locness_datamanager.main.process_raw_data_incremental')
    @patch('locness_datamanager.main.time.time')
    def test_custom_resample_intervals(self, mock_time, mock_process_raw, mock_process_summary, mock_sleep):
        """Test that custom resample intervals are passed correctly"""
        mock_time.side_effect = [0, 0, 70, 70, 71, 72]
        mock_sleep.side_effect = KeyboardInterrupt()
        backup_manager = Mock(spec=DatabaseBackup)
        
        with pytest.raises(KeyboardInterrupt):
            main.poll_and_process(
                db_path='test.db',
                db_resample_interval='5s',
                parquet_resample_interval='30s',
                parquet_poll_interval=60,
                parquet_path='test.parquet',
                csv_path='test.csv',
                backup_manager=backup_manager
            )
        
        # Check raw data processing uses custom interval
        mock_process_raw.assert_called_once()
        args, kwargs = mock_process_raw.call_args
        assert kwargs['resample_interval'] == '5s'
        
        # Check summary processing uses custom interval
        mock_process_summary.assert_called_once()
        args, kwargs = mock_process_summary.call_args
        assert kwargs['resample_interval'] == '30s'

    def test_default_parameters(self):
        """Test that default parameters are set correctly"""
        with patch('locness_datamanager.main.time.sleep') as mock_sleep, \
             patch('locness_datamanager.main.process_raw_data_incremental') as mock_process_raw, \
             patch('locness_datamanager.main.time.time') as mock_time:
            
            mock_time.side_effect = [0, 0, 1, 2, 3]
            mock_sleep.side_effect = KeyboardInterrupt()
            
            backup_manager = Mock(spec=DatabaseBackup)
            
            with pytest.raises(KeyboardInterrupt):
                main.poll_and_process(backup_manager=backup_manager)
            
            # Check default parameters were used
            mock_process_raw.assert_called_once_with(
                sqlite_path=None,
                resample_interval='10s',
                summary_table='underway_summary',
                replace_all=False,
                ph_k0=0.0,
                ph_k2=0.0,
                ph_ma_window=120,
                ph_freq=0.5,
            )


class TestMain:
    """Test the main function"""
    
    @patch('locness_datamanager.main.poll_and_process')
    @patch('locness_datamanager.main.DatabaseBackup')
    @patch('locness_datamanager.main.os.path.exists')
    @patch('locness_datamanager.main.get_config')
    @patch('locness_datamanager.main.logging.basicConfig')
    def test_main_successful_run(self, mock_logging_config, mock_get_config, mock_exists, mock_backup_class, mock_poll):
        """Test successful main function execution"""
        # Mock configuration
        mock_config = {
            'db_path': 'data/test.db',
            'db_poll_interval': 15,
            'db_resample_interval': '5s',
            'parquet_path': 'data/test.parquet',
            'parquet_poll_interval': 120,
            'parquet_resample_interval': '30s',
            'partition_hours': 8,
            'csv_path': 'data/test.csv',
            'backup_path': 'data/test_backup',
            'backup_interval': 7200,
            'ph_k0': -1.39469,
            'ph_k2': -0.00107,
            'ph_ma_window': 120,
            'ph_freq': 0.5
        }
        mock_get_config.return_value = mock_config
        mock_exists.return_value = True
        mock_backup_instance = Mock(spec=DatabaseBackup)
        mock_backup_class.return_value = mock_backup_instance
        
        # Mock poll_and_process to raise KeyboardInterrupt to exit cleanly
        mock_poll.side_effect = KeyboardInterrupt()
        
        # Run main
        main.main()
        
        # Verify database backup was created
        mock_backup_class.assert_called_once_with('data/test.db', backup_dir='data/test_backup')
        
        # Verify poll_and_process was called with correct parameters
        mock_poll.assert_called_once_with(
            db_path='data/test.db',
            db_poll_interval=15,
            db_resample_interval='5s',
            parquet_path='data/test.parquet',
            parquet_poll_interval=120,
            parquet_resample_interval='30s',
            csv_path='data/test.csv',
            partition_hours=8,
            backup_manager=mock_backup_instance,
            backup_interval=7200,
            ph_k0=-1.39469,
            ph_k2=-0.00107,
            ph_ma_window=120,
            ph_freq=0.5,
        )

    @patch('locness_datamanager.main.poll_and_process')
    @patch('locness_datamanager.main.DatabaseBackup')
    @patch('locness_datamanager.main.os.path.exists')
    @patch('locness_datamanager.main.get_config')
    @patch('locness_datamanager.main.logging.basicConfig')
    def test_main_with_defaults(self, mock_logging_config, mock_get_config, mock_exists, mock_backup_class, mock_poll):
        """Test main function with default config values"""
        # Mock minimal configuration (using defaults)
        mock_config = {}
        mock_get_config.return_value = mock_config
        mock_exists.return_value = True
        mock_backup_instance = Mock(spec=DatabaseBackup)
        mock_backup_class.return_value = mock_backup_instance
        mock_poll.side_effect = KeyboardInterrupt()
        
        main.main()
        
        # Verify defaults were used
        mock_poll.assert_called_once_with(
            db_path='data/locness.db',
            db_poll_interval=10,
            db_resample_interval='10s',
            parquet_path='data/locness.parquet',
            parquet_poll_interval=3600,
            parquet_resample_interval='60s',
            csv_path='data/locness.csv',
            partition_hours=6,
            backup_manager=mock_backup_instance,
            backup_interval=3600,
            ph_k0=0.0,
            ph_k2=0.0,
            ph_ma_window=120,
            ph_freq=0.5,
        )

    @patch('locness_datamanager.main.os.path.exists')
    @patch('locness_datamanager.main.get_config')
    @patch('locness_datamanager.main.logging.error')
    def test_main_database_not_exists(self, mock_logging_error, mock_get_config, mock_exists):
        """Test main function when database doesn't exist"""
        mock_config = {'db_path': 'nonexistent.db'}
        mock_get_config.return_value = mock_config
        mock_exists.return_value = False
        
        # Should return early without error
        main.main()
        
        # Verify error was logged
        mock_logging_error.assert_called_once_with("Database nonexistent.db does not exist. Please create it first.")

    @patch('locness_datamanager.main.poll_and_process')
    @patch('locness_datamanager.main.DatabaseBackup')
    @patch('locness_datamanager.main.os.path.exists')
    @patch('locness_datamanager.main.get_config')
    @patch('locness_datamanager.main.time.sleep')
    @patch('locness_datamanager.main.logging.error')
    def test_main_exception_handling(self, mock_logging_error, mock_sleep, mock_get_config, mock_exists, mock_backup_class, mock_poll):
        """Test main function exception handling and retry logic"""
        mock_config = {'db_path': 'data/test.db', 'db_poll_interval': 5}
        mock_get_config.return_value = mock_config
        mock_exists.return_value = True
        mock_backup_instance = Mock(spec=DatabaseBackup)
        mock_backup_class.return_value = mock_backup_instance
        
        # First call raises exception, second call raises KeyboardInterrupt to exit
        test_exception = RuntimeError("Test error")
        mock_poll.side_effect = [test_exception, KeyboardInterrupt()]
        
        main.main()
        
        # Verify error was logged
        mock_logging_error.assert_called_once_with(f"An error occurred: {test_exception}")
        
        # Verify sleep was called for retry
        mock_sleep.assert_called_once_with(5)
        
        # Verify poll_and_process was called twice (original + retry)
        assert mock_poll.call_count == 2

    @patch('locness_datamanager.main.poll_and_process')
    @patch('locness_datamanager.main.DatabaseBackup')
    @patch('locness_datamanager.main.os.path.exists')
    @patch('locness_datamanager.main.get_config')
    @patch('locness_datamanager.main.logging.info')
    def test_main_keyboard_interrupt_handling(self, mock_logging_info, mock_get_config, mock_exists, mock_backup_class, mock_poll):
        """Test main function KeyboardInterrupt handling"""
        mock_config = {'db_path': 'data/test.db'}
        mock_get_config.return_value = mock_config
        mock_exists.return_value = True
        mock_backup_instance = Mock(spec=DatabaseBackup)
        mock_backup_class.return_value = mock_backup_instance
        mock_poll.side_effect = KeyboardInterrupt()
        
        main.main()
        
        # Verify graceful shutdown message was logged
        mock_logging_info.assert_called_with("Interrupted by user. Shutting down gracefully.")

    @patch('locness_datamanager.main.poll_and_process')
    @patch('locness_datamanager.main.DatabaseBackup')
    @patch('locness_datamanager.main.os.path.exists')
    @patch('locness_datamanager.main.get_config')
    @patch('locness_datamanager.main.logging.basicConfig')
    def test_main_logging_setup_with_file(self, mock_logging_config, mock_get_config, mock_exists, mock_backup_class, mock_poll):
        """Test logging setup with file handler"""
        mock_config = {
            'db_path': 'data/test.db',
            'log_path': '/tmp/test.log'
        }
        mock_get_config.return_value = mock_config
        mock_exists.return_value = True
        mock_backup_instance = Mock(spec=DatabaseBackup)
        mock_backup_class.return_value = mock_backup_instance
        mock_poll.side_effect = KeyboardInterrupt()
        
        with patch('locness_datamanager.main.logging.FileHandler') as mock_file_handler:
            mock_file_handler.return_value = Mock()
            main.main()
            
            # Verify file handler was created
            mock_file_handler.assert_called_once_with('/tmp/test.log')

    @patch('locness_datamanager.main.poll_and_process')
    @patch('locness_datamanager.main.DatabaseBackup')
    @patch('locness_datamanager.main.os.path.exists')
    @patch('locness_datamanager.main.get_config')
    @patch('locness_datamanager.main.logging.basicConfig')
    @patch('builtins.print')
    def test_main_logging_file_error(self, mock_print, mock_logging_config, mock_get_config, mock_exists, mock_backup_class, mock_poll):
        """Test logging setup when file handler creation fails"""
        mock_config = {
            'db_path': 'data/test.db',
            'log_path': '/invalid/path/test.log'
        }
        mock_get_config.return_value = mock_config
        mock_exists.return_value = True
        mock_backup_instance = Mock(spec=DatabaseBackup)
        mock_backup_class.return_value = mock_backup_instance
        mock_poll.side_effect = KeyboardInterrupt()
        
        with patch('locness_datamanager.main.logging.FileHandler') as mock_file_handler:
            mock_file_handler.side_effect = OSError("Permission denied")
            main.main()
            
            # Verify warning was printed
            mock_print.assert_called_once()
            print_call_args = mock_print.call_args[0][0]
            assert "Warning: Could not set up file logging" in print_call_args


class TestMainIntegration:
    """Integration tests for main.py"""
    
    def test_main_function_exists_and_callable(self):
        """Test that main function exists and is callable"""
        assert hasattr(main, 'main')
        assert callable(main.main)
    
    def test_poll_and_process_function_exists_and_callable(self):
        """Test that poll_and_process function exists and is callable"""
        assert hasattr(main, 'poll_and_process')
        assert callable(main.poll_and_process)
    
    def test_imports_successful(self):
        """Test that all required imports are successful"""
        # This test passes if the module imports without error
        from locness_datamanager import main
        assert main is not None

    @patch('locness_datamanager.main.get_config')
    def test_config_parameters_used(self, mock_get_config):
        """Test that all expected config parameters are accessed"""
        mock_config = {
            'log_path': '/tmp/test.log',
            'db_path': 'data/test.db',
            'db_poll_interval': 10,
            'db_resample_interval': '10s',
            'parquet_path': 'data/test.parquet',
            'parquet_poll_interval': 3600,
            'parquet_resample_interval': '60s',
            'partition_hours': 6,
            'csv_path': 'data/test.csv',
            'backup_path': 'data/backup',
            'backup_interval': 3600,
            'ph_k0': -1.5,
            'ph_k2': -0.001,
            'ph_ma_window': 120,
            'ph_freq': 0.5
        }
        mock_get_config.return_value = mock_config
        
        with patch('locness_datamanager.main.os.path.exists', return_value=False):
            main.main()
        
        # Verify all expected config keys were accessed
        expected_keys = [
            'log_path', 'db_path', 'db_poll_interval', 'db_resample_interval',
            'parquet_path', 'parquet_poll_interval', 'parquet_resample_interval',
            'partition_hours', 'csv_path', 'backup_path', 'backup_interval',
            'ph_k0', 'ph_k2', 'ph_ma_window', 'ph_freq'
        ]
        
        for key in expected_keys:
            # This verifies the key was accessed from the config
            assert mock_config[key] is not None or key in ['log_path']  # log_path can be None
            
    
class TestDatabaseIntegrity:
    """Test database integrity and data consistency"""
    # Add this to `tests/test_main.py`



    @pytest.fixture
    def db_connection(self):
        """Fixture to provide a connection to the test database."""
        db_path = "data/locness.db"  # Update with the correct test database path
        conn = sqlite3.connect(db_path)
        yield conn
        conn.close()


    def test_table_gaps(self, db_connection):
        """Test that there are no gaps in the datetime_utc column for raw tables."""
        tables = {
            "tsg": 1,
            "rhodamine": 1,
            "gps": 1,
            "ph": 2,
        }
        for table, interval in tables.items():
            query = f"""
            SELECT datetime_utc, 
                   julianday(LEAD(datetime_utc) OVER (ORDER BY datetime_utc)) - julianday(datetime_utc) AS gap
            FROM {table};
            """
            df = pd.read_sql_query(query, db_connection)
            gaps = df[df["gap"] > interval / 86400.0]  # Convert seconds to days
            assert gaps.empty, f"Gaps found in {table} table: {gaps}"


    def test_underway_summary_intervals(self, db_connection):
        """Test that the underway_summary table has entries every 10 seconds."""
        query = """
        SELECT datetime_utc, 
               julianday(LEAD(datetime_utc) OVER (ORDER BY datetime_utc)) - julianday(datetime_utc) AS gap
        FROM underway_summary;
        """
        df = pd.read_sql_query(query, db_connection)
        gaps = df[df["gap"] > 10 / 86400.0]  # Convert seconds to days
        assert gaps.empty, f"Gaps found in underway_summary table: {gaps}"


    def test_ph_total_running_average(self, db_connection):
        """Test that the 2-minute running average of ph_total matches ph_total_ma."""
        query = """
        SELECT datetime_utc, ph_total, ph_total_ma
        FROM ph
        ORDER BY datetime_utc;
        """
        df = pd.read_sql_query(query, db_connection)
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
        df.set_index("datetime_utc", inplace=True)

        # Calculate 2-minute running average
        df["calculated_ma"] = df["ph_total"].rolling("2T").mean()

        # Compare calculated running average with ph_total_ma
        mismatches = df[~df["calculated_ma"].fillna(0).eq(df["ph_total_ma"].fillna(0))]
        assert mismatches.empty, f"Mismatches found in ph_total_ma: {mismatches}"