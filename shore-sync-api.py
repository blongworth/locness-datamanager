from googleapiclient.errors import HttpError
from googleapiclient.discovery import build as gsheet_build
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import tempfile
import argparse
import shutil
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pyarrow as pa
import pyarrow.parquet as pq

# Google Drive API scope
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


class GoogleDriveFileHandler:
    def __init__(self, credentials_path="credentials.json", shared_drive_id=None):
        self.credentials_path = credentials_path
        self.shared_drive_id = shared_drive_id
        self.service = None
        self.sheets_service = None
        self.drive_service = None
        self.authenticate()

    def get_sheet_id(self, sheet_ref):
        """
        Accepts a Google Sheet name, ID, or full URL. Returns the spreadsheetId if accessible, else None.
        """
        # If it's a URL, extract the ID
        if sheet_ref.startswith("http"):
            import re
            match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_ref)
            if match:
                return match.group(1)
            else:
                print(f"Could not extract spreadsheetId from URL: {sheet_ref}")
                return None
        # If it's a 44-char ID, just return it
        if len(sheet_ref) >= 30 and len(sheet_ref) <= 80 and all(c.isalnum() or c in '-_' for c in sheet_ref):
            return sheet_ref
        # Otherwise, treat as a name and search
        try:
            results = self.drive_service.files().list(
                q=f"name='{sheet_ref}' and mimeType='application/vnd.google-apps.spreadsheet'",
                fields="files(id, name)",
            ).execute()
            files = results.get("files", [])
            if files:
                return files[0]["id"]
            return None
        except Exception as e:
            print(f"Error searching for Google Sheet: {e}")
            return None

    def create_sheet(self, sheet_name, header):
        # Create a new Google Sheet and write the header row
        spreadsheet = {
            "properties": {"title": sheet_name}
        }
        sheet = self.sheets_service.spreadsheets().create(body=spreadsheet, fields="spreadsheetId").execute()
        sheet_id = sheet["spreadsheetId"]
        self.append_rows(sheet_id, [header])
        print(f"Created new Google Sheet: {sheet_name} (ID: {sheet_id})")
        return sheet_id

    def append_rows(self, sheet_id, rows):
        # Append rows to the first sheet
        body = {"values": rows}
        try:
            self.sheets_service.spreadsheets().values().append(
                spreadsheetId=sheet_id,
                range="A1",
                valueInputOption="RAW",
                insertDataOption="INSERT_ROWS",
                body=body,
            ).execute()
        except HttpError as e:
            print(f"Error appending to Google Sheet: {e}")

    def __init__(self, credentials_path="credentials.json", shared_drive_id=None):
        self.credentials_path = credentials_path
        self.shared_drive_id = shared_drive_id
        self.service = None
        self.authenticate()

    def authenticate(self):
        """Authenticate with Google Drive and Google Sheets APIs using service account only."""
        from google.oauth2 import service_account
        try:
            # Use a superset of scopes for both Drive and Sheets
            scopes = [
                "https://www.googleapis.com/auth/drive.file",
                "https://www.googleapis.com/auth/spreadsheets",
            ]
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=scopes
            )
            self.service = build("drive", "v3", credentials=creds)
            self.sheets_service = gsheet_build("sheets", "v4", credentials=creds)
            self.drive_service = build("drive", "v3", credentials=creds)
            print("Authenticated with Google Drive and Google Sheets using service account.")
        except Exception as e:
            print(f"Error reading service account credentials: {e}")
            raise

    def list_shared_drives(self):
        try:
            results = self.service.drives().list().execute()
            drives = results.get("drives", [])
            if not drives:
                print("No shared drives found.")
                return []
            print("Available shared drives:")
            for drive in drives:
                print(f"  Name: {drive['name']}")
                print(f"  ID: {drive['id']}")
                print(f"  ---")
            return drives
        except Exception as e:
            print(f"Error listing shared drives: {e}")
            return []

    def list_files_in_shared_drive(self, folder_name=None):
        if not self.shared_drive_id:
            print("No shared drive ID specified")
            return []
        try:
            query = f"'{self.shared_drive_id}' in parents"
            if folder_name:
                folder_results = (
                    self.service.files()
                    .list(
                        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
                        driveId=self.shared_drive_id,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                        corpora="drive",
                        fields="files(id, name, parents)",
                    )
                    .execute()
                )
                folders = folder_results.get("files", [])
                if folders:
                    folder_id = folders[0]["id"]
                    query = f"'{folder_id}' in parents"
                else:
                    print(f"Folder '{folder_name}' not found")
                    return []
            results = (
                self.service.files()
                .list(
                    q=query,
                    driveId=self.shared_drive_id,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    corpora="drive",
                    fields="files(id, name, mimeType, modifiedTime)",
                )
                .execute()
            )
            files = results.get("files", [])
            if not files:
                location = f"folder '{folder_name}'" if folder_name else "shared drive"
                print(f"No files found in {location}")
                return []
            print(
                f"Files in {'folder ' + folder_name if folder_name else 'shared drive'}:"
            )
            for file in files:
                print(f"  Name: {file['name']}")
                print(f"  ID: {file['id']}")
                print(f"  Type: {file['mimeType']}")
                print(f"  Modified: {file.get('modifiedTime', 'Unknown')}")
                print(f"  ---")
            return files
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def upload_to_drive(self, file_path, filename, folder_name=None):
        try:
            file_metadata = {"name": filename}
            if self.shared_drive_id:
                file_metadata["parents"] = [self.shared_drive_id]
                if folder_name:
                    folder_id = self._find_or_create_folder(folder_name)
                    if folder_id:
                        file_metadata["parents"] = [folder_id]
                search_query = f"name='{filename}'"
                if folder_name:
                    folder_id = self._find_or_create_folder(folder_name)
                    if folder_id:
                        search_query = f"name='{filename}' and '{folder_id}' in parents"
                results = (
                    self.service.files()
                    .list(
                        q=search_query,
                        driveId=self.shared_drive_id,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                        corpora="drive",
                        fields="files(id, name)",
                    )
                    .execute()
                )
            else:
                results = (
                    self.service.files()
                    .list(q=f"name='{filename}'", fields="files(id, name)")
                    .execute()
                )
            files = results.get("files", [])
            media = MediaFileUpload(file_path, mimetype="application/octet-stream")
            if files:
                file_id = files[0]["id"]
                file = (
                    self.service.files()
                    .update(fileId=file_id, media_body=media, supportsAllDrives=True)
                    .execute()
                )
                print(f"Updated existing file: {filename}")
            else:
                file = (
                    self.service.files()
                    .create(
                        body=file_metadata,
                        media_body=media,
                        supportsAllDrives=True,
                        fields="id",
                    )
                    .execute()
                )
                print(f"Created new file: {filename}")
            return file.get("id")
        except Exception as e:
            print(f"Error uploading file: {str(e)}")
            return None

    def _find_or_create_folder(self, folder_name):
        if not self.shared_drive_id:
            return None
        try:
            results = (
                self.service.files()
                .list(
                    q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
                    driveId=self.shared_drive_id,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    corpora="drive",
                    fields="files(id, name)",
                )
                .execute()
            )
            folders = results.get("files", [])
            if folders:
                return folders[0]["id"]
            folder_metadata = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [self.shared_drive_id],
            }
            folder = (
                self.service.files()
                .create(body=folder_metadata, supportsAllDrives=True, fields="id")
                .execute()
            )
            print(f"Created folder: {folder_name}")
            return folder.get("id")
        except Exception as e:
            print(f"Error finding/creating folder: {e}")
            return None


class SyntheticDataGenerator:
    @staticmethod
    def generate(n_records=1):
        """
        Generate synthetic oceanographic data.
        Args:
            n_records: Number of records to generate
        Returns:
            pandas.DataFrame with synthetic data
        """
        base_lat = 40.7128
        base_lon = -74.0060
        data = []
        current_time = datetime.now()
        for i in range(n_records):
            record = {
                "timestamp": current_time + timedelta(seconds=i),
                "lat": base_lat + np.random.normal(0, 0.01),
                "lon": base_lon + np.random.normal(0, 0.01),
                "temp": np.random.normal(15.0, 3.0),
                "salinity": np.random.normal(35.0, 2.0),
                "ph": np.random.normal(8.1, 0.2),
                "rhodamine": np.random.exponential(0.5),
            }
            record["temp"] = max(-2, min(30, record["temp"]))
            record["salinity"] = max(30, min(40, record["salinity"]))
            record["ph"] = max(7.5, min(8.5, record["ph"]))
            record["rhodamine"] = max(0, record["rhodamine"])
            data.append(record)
        return pd.DataFrame(data)


def run_continuous_generation(
    file_handler,
    filenames=["synthetic_oceanographic_data.parquet"],
    interval_seconds=60,
    records_per_batch=10,
    folder_name=None,
):
    print(f"Starting continuous data generation...")
    print(f"Files: {', '.join(filenames)}")
    if folder_name:
        print(f"Folder: {folder_name}")
    if file_handler.shared_drive_id:
        print(f"Shared Drive ID: {file_handler.shared_drive_id}")
    print(f"Interval: {interval_seconds} seconds")
    print(f"Records per batch: {records_per_batch}")
    print("Press Ctrl+C to stop")
    sheet_ref = None
    for f in filenames:
        if f.lower().endswith(".gsheet") or f.lower().endswith(".gsheets") or f.startswith("http") or (len(f) >= 30 and len(f) <= 80 and all(c.isalnum() or c in '-_' for c in f)):
            sheet_ref = f
            break
    sheets_handler = file_handler
    try:
        while True:
            new_data = SyntheticDataGenerator.generate(records_per_batch)
            # Write to CSV/Parquet as before
            for filename in filenames:
                if filename.lower().endswith(".csv"):
                    if not os.path.exists(filename):
                        new_data.to_csv(filename, index=False)
                    else:
                        new_data.to_csv(filename, mode="a", header=False, index=False)
                    tmp_suffix = ".csv"
                elif filename.lower().endswith(".parquet"):
                    if not os.path.exists(filename):
                        new_data.to_parquet(filename, index=False)
                    else:
                        table = pa.Table.from_pandas(new_data)
                        with pq.ParquetWriter(filename + '.tmp', table.schema) as writer:
                            for batch in pq.ParquetFile(filename).iter_batches():
                                writer.write_table(pa.Table.from_batches([batch]))
                            writer.write_table(table)
                        os.replace(filename + '.tmp', filename)
                    tmp_suffix = ".parquet"
                elif filename.lower().endswith(".gsheet") or filename.lower().endswith(".gsheets") or filename.startswith("http") or (len(filename) >= 30 and len(filename) <= 80 and all(c.isalnum() or c in '-_' for c in filename)):
                    # Google Sheets logic
                    sheet_id = sheets_handler.get_sheet_id(filename)
                    # Convert all datetime columns to ISO strings for Sheets
                    df_for_sheet = new_data.copy()
                    for col in df_for_sheet.columns:
                        if pd.api.types.is_datetime64_any_dtype(df_for_sheet[col]) or pd.api.types.is_timedelta64_dtype(df_for_sheet[col]) or pd.api.types.is_object_dtype(df_for_sheet[col]):
                            # Try to convert Timestamps to string if possible
                            df_for_sheet[col] = df_for_sheet[col].apply(lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x))
                    rows = df_for_sheet.values.tolist()
                    header = list(df_for_sheet.columns)
                    if not sheet_id:
                        print(f"Could not resolve Google Sheet ID for: {filename}")
                        continue
                    # Try to append, if fails with 400, maybe sheet is empty, so try to write header first
                    try:
                        sheets_handler.append_rows(sheet_id, rows)
                        print(f"Appended {len(rows)} new records to Google Sheet: {filename}")
                    except Exception as e:
                        print(f"Error appending rows, trying to write header first: {e}")
                        try:
                            sheets_handler.append_rows(sheet_id, [header])
                            sheets_handler.append_rows(sheet_id, rows)
                            print(f"Wrote header and appended {len(rows)} new records to Google Sheet: {filename}")
                        except Exception as e2:
                            print(f"Failed to write to Google Sheet: {e2}")
                    continue
                else:
                    continue
                # Upload to Google Drive for CSV/Parquet
                with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp_file:
                    tmp_path = tmp_file.name
                shutil.copyfile(filename, tmp_path)
                file_id = file_handler.upload_to_drive(tmp_path, filename, folder_name)
                os.unlink(tmp_path)
                if file_id:
                    print(f"Successfully uploaded {len(new_data)} new records to {filename} at {datetime.now()}")
                else:
                    print(f"Failed to upload data to {filename}")
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nStopping data generation...")
    except Exception as e:
        print(f"Error in continuous generation: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic oceanographic data and upload to Google Drive"
    )
    parser.add_argument(
        "--basename",
        default="synthetic_oceanographic_data",
        help="Base name for CSV and Parquet files (no extension)",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Google Sheet URL, ID, or name to write to (optional)",
    )
    parser.add_argument(
        "--interval", type=int, default=60, help="Interval between uploads in seconds"
    )
    parser.add_argument(
        "--records",
        type=int,
        default=60,
        help="Number of records to generate per batch",
    )
    parser.add_argument(
        "--credentials",
        default="credentials.json",
        help="Path to Google API credentials file",
    )
    # --token argument removed; not needed for service account
    parser.add_argument(
        "--shared-drive-id", help="ID of the shared drive (required for shared drives)"
    )
    parser.add_argument(
        "--folder", help="Folder name within the shared drive (optional)"
    )
    parser.add_argument(
        "--list-drives",
        action="store_true",
        help="List all accessible shared drives and exit",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List files in the shared drive and exit",
    )

    args = parser.parse_args()

    # Check if credentials file exists
    if not os.path.exists(args.credentials):
        print(f"Error: Credentials file '{args.credentials}' not found.")
        print(
            "Please download your Google API credentials and save as 'credentials.json'"
        )
        print("Visit: https://console.cloud.google.com/apis/credentials")
        return

    # Initialize generator
    file_handler = GoogleDriveFileHandler(
        args.credentials, args.shared_drive_id
    )

    # Handle utility commands
    if args.list_drives:
        file_handler.list_shared_drives()
        return

    if args.list_files:
        if not args.shared_drive_id:
            print("Error: --shared-drive-id is required when using --list-files")
            return
        file_handler.list_files_in_shared_drive(args.folder)
        return

    # For shared drives, require shared drive ID
    if args.shared_drive_id is None:
        print("Warning: No shared drive ID specified. Using personal Google Drive.")
        print("To use a shared drive, first run with --list-drives to find the ID")
        print("Then run with --shared-drive-id <ID>")

    # Run continuous generation
    # Support uploading to both CSV and Parquet if desired
    filenames = [f"{args.basename}.csv", f"{args.basename}.parquet"]
    if args.sheet:
        filenames.append(args.sheet)
    run_continuous_generation(
        file_handler=file_handler,
        filenames=filenames,
        interval_seconds=args.interval,
        records_per_batch=args.records,
        folder_name=args.folder,
    )


if __name__ == "__main__":
    main()
