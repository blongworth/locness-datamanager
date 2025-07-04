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
        self.authenticate()

    def authenticate(self):
        """Authenticate with Google Drive API using service account only."""
        from google.oauth2 import service_account
        try:
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=SCOPES
            )
            self.service = build("drive", "v3", credentials=creds)
            print("Authenticated with Google Drive using service account.")
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
    filename="synthetic_oceanographic_data.parquet",
    interval_seconds=60,
    records_per_batch=10,
    folder_name=None,
):
    print(f"Starting continuous data generation...")
    print(f"File: {filename}")
    if folder_name:
        print(f"Folder: {folder_name}")
    if file_handler.shared_drive_id:
        print(f"Shared Drive ID: {file_handler.shared_drive_id}")
    print(f"Interval: {interval_seconds} seconds")
    print(f"Records per batch: {records_per_batch}")
    print("Press Ctrl+C to stop")
    try:
        while True:
            new_data = SyntheticDataGenerator.generate(records_per_batch)
            if not os.path.exists(filename):
                # Create new file with the new data
                new_data.to_parquet(filename, index=False)
            else:
                # Append new data to the existing file without reading all data into memory
                # Parquet does not support append natively, so we use a workaround
                table = pa.Table.from_pandas(new_data)
                with pq.ParquetWriter(filename + '.tmp', table.schema) as writer:
                    for batch in pq.ParquetFile(filename).iter_batches():
                        writer.write_table(pa.Table.from_batches([batch]))
                    writer.write_table(table)
                os.replace(filename + '.tmp', filename)
            # Upload to Google Drive
            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
                tmp_path = tmp_file.name
            shutil.copyfile(filename, tmp_path)
            file_id = file_handler.upload_to_drive(tmp_path, filename, folder_name)
            os.unlink(tmp_path)
            if file_id:
                print(f"Successfully uploaded {len(new_data)} new records at {datetime.now()}")
            else:
                print("Failed to upload data")
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
        "--filename",
        default="synthetic_oceanographic_data.parquet",
        help="Name of the parquet file on Google Drive",
    )
    parser.add_argument(
        "--interval", type=int, default=60, help="Interval between uploads in seconds"
    )
    parser.add_argument(
        "--records",
        type=int,
        default=10,
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
    run_continuous_generation(
        file_handler=file_handler,
        filename=args.filename,
        interval_seconds=args.interval,
        records_per_batch=args.records,
        folder_name=args.folder,
    )


if __name__ == "__main__":
    main()
