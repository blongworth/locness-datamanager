import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pickle
import tempfile
import argparse

# Google Drive API scope
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


class SyntheticDataGenerator:
    def __init__(
        self,
        credentials_path="credentials.json",
        token_path="token.pickle",
        shared_drive_id=None,
    ):
        """
        Initialize the synthetic data generator.

        Args:
            credentials_path: Path to Google API credentials JSON file
            token_path: Path to store authentication token
            shared_drive_id: ID of the shared drive (if using shared drives)
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.shared_drive_id = shared_drive_id
        self.service = None
        self.authenticate()

    def list_shared_drives(self):
        """List all shared drives accessible to the service account"""
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
        """List files in the shared drive, optionally in a specific folder"""
        if not self.shared_drive_id:
            print("No shared drive ID specified")
            return []

        try:
            query = f"'{self.shared_drive_id}' in parents"
            if folder_name:
                # First find the folder
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

    def authenticate(self):
        """Authenticate with Google Drive API"""
        creds = None

        # Check if it's a service account file (contains "type": "service_account")
        try:
            import json

            with open(self.credentials_path, "r") as f:
                cred_data = json.load(f)

            if cred_data.get("type") == "service_account":
                # Use service account authentication
                from google.oauth2 import service_account

                creds = service_account.Credentials.from_service_account_file(
                    self.credentials_path, scopes=SCOPES
                )
                print("Using service account authentication")
            else:
                # Use OAuth2 flow for user credentials
                self._oauth_authenticate()
                return

        except Exception as e:
            print(f"Error reading credentials file: {e}")
            self._oauth_authenticate()
            return

        self.service = build("drive", "v3", credentials=creds)
        print("Successfully authenticated with Google Drive")

    def _oauth_authenticate(self):
        """Handle OAuth2 authentication flow"""
        creds = None

        # Load existing token if available
        if os.path.exists(self.token_path):
            with open(self.token_path, "rb") as token:
                creds = pickle.load(token)

        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES
                )

                print("Attempting OAuth authentication...")
                print(
                    "If you get a redirect_uri_mismatch error, please add these URIs to your Google Cloud Console:"
                )
                print("  - http://localhost:8080/")
                print("  - http://127.0.0.1:8080/")

                # Try different approaches for OAuth flow
                try:
                    # First try with a specific port and no browser opening
                    creds = flow.run_local_server(
                        port=8080,
                        open_browser=False,
                        authorization_prompt_message="Please visit this URL to authorize the application:\n{url}",
                    )
                except Exception as e:
                    print(f"Local server on port 8080 failed: {e}")
                    try:
                        # Try port 8081 as backup
                        print("Trying port 8081...")
                        creds = flow.run_local_server(
                            port=8081,
                            open_browser=False,
                            authorization_prompt_message="Please visit this URL to authorize the application:\n{url}",
                        )
                    except Exception as e2:
                        print(f"Local server on port 8081 also failed: {e2}")
                        print("Falling back to console-based authentication...")
                        print(
                            "You'll need to manually copy/paste the authorization code."
                        )
                        # Fallback to console-based flow
                        creds = flow.run_console()

            # Save credentials for next run
            with open(self.token_path, "wb") as token:
                pickle.dump(creds, token)

        self.service = build("drive", "v3", credentials=creds)

    def generate_synthetic_data(self, n_records=1):
        """
        Generate synthetic oceanographic data.

        Args:
            n_records: Number of records to generate

        Returns:
            pandas.DataFrame with synthetic data
        """
        # Base location (example: somewhere in the Atlantic)
        base_lat = 40.7128
        base_lon = -74.0060

        data = []
        current_time = datetime.now()

        for i in range(n_records):
            # Generate realistic oceanographic data
            record = {
                "timestamp": current_time + timedelta(seconds=i),
                "lat": base_lat
                + np.random.normal(0, 0.01),  # Small variations around base
                "lon": base_lon + np.random.normal(0, 0.01),
                "temp": np.random.normal(15.0, 3.0),  # Temperature in Celsius
                "salinity": np.random.normal(35.0, 2.0),  # Salinity in PSU
                "ph": np.random.normal(8.1, 0.2),  # pH levels
                "rhodamine": np.random.exponential(0.5),  # Rhodamine concentration
            }

            # Ensure realistic bounds
            record["temp"] = max(-2, min(30, record["temp"]))
            record["salinity"] = max(30, min(40, record["salinity"]))
            record["ph"] = max(7.5, min(8.5, record["ph"]))
            record["rhodamine"] = max(0, record["rhodamine"])

            data.append(record)

        return pd.DataFrame(data)

    def upload_to_drive(self, file_path, filename, folder_name=None):
        """
        Upload file to Google Drive or Shared Drive.

        Args:
            file_path: Local path to the file
            filename: Name for the file on Google Drive
            folder_name: Optional folder name within the shared drive

        Returns:
            File ID of uploaded file
        """
        try:
            file_metadata = {"name": filename}

            # Handle shared drive uploads
            if self.shared_drive_id:
                file_metadata["parents"] = [self.shared_drive_id]

                # If folder specified, find/create it in the shared drive
                if folder_name:
                    folder_id = self._find_or_create_folder(folder_name)
                    if folder_id:
                        file_metadata["parents"] = [folder_id]

                # Check if file already exists in shared drive
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
                # Regular Google Drive search
                results = (
                    self.service.files()
                    .list(q=f"name='{filename}'", fields="files(id, name)")
                    .execute()
                )

            files = results.get("files", [])
            media = MediaFileUpload(file_path, mimetype="application/octet-stream")

            if files:
                # Update existing file
                file_id = files[0]["id"]
                file = (
                    self.service.files()
                    .update(fileId=file_id, media_body=media, supportsAllDrives=True)
                    .execute()
                )
                print(f"Updated existing file: {filename}")
            else:
                # Create new file
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
        """Find or create a folder in the shared drive"""
        if not self.shared_drive_id:
            return None

        try:
            # Search for existing folder
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

            # Create folder if it doesn't exist
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

    def run_continuous_generation(
        self,
        filename="synthetic_oceanographic_data.parquet",
        interval_seconds=60,
        records_per_batch=10,
        folder_name=None,
    ):
        """
        Continuously generate and upload synthetic data.

        Args:
            filename: Name of the parquet file on Google Drive
            interval_seconds: Time between data generation cycles
            records_per_batch: Number of records to generate per batch
            folder_name: Optional folder name within the shared drive
        """
        print(f"Starting continuous data generation...")
        print(f"File: {filename}")
        if folder_name:
            print(f"Folder: {folder_name}")
        if self.shared_drive_id:
            print(f"Shared Drive ID: {self.shared_drive_id}")
        print(f"Interval: {interval_seconds} seconds")
        print(f"Records per batch: {records_per_batch}")
        print("Press Ctrl+C to stop")

        # Initialize with empty DataFrame
        all_data = pd.DataFrame()

        try:
            while True:
                # Generate new data
                new_data = self.generate_synthetic_data(records_per_batch)

                # Append to existing data
                all_data = pd.concat([all_data, new_data], ignore_index=True)

                # Keep only last 1000 records to prevent file from growing too large
                if len(all_data) > 1000:
                    all_data = all_data.tail(1000)

                # Create temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".parquet"
                ) as tmp_file:
                    tmp_path = tmp_file.name

                # Write to parquet
                all_data.to_parquet(tmp_path, index=False)

                # Upload to Google Drive
                file_id = self.upload_to_drive(tmp_path, filename, folder_name)

                # Clean up temporary file
                os.unlink(tmp_path)

                if file_id:
                    print(
                        f"Successfully uploaded {len(new_data)} new records at {datetime.now()}"
                    )
                    print(f"Total records in file: {len(all_data)}")
                else:
                    print("Failed to upload data")

                # Wait for next cycle
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
    parser.add_argument(
        "--token", default="token.pickle", help="Path to store authentication token"
    )
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
    generator = SyntheticDataGenerator(
        args.credentials, args.token, args.shared_drive_id
    )

    # Handle utility commands
    if args.list_drives:
        generator.list_shared_drives()
        return

    if args.list_files:
        if not args.shared_drive_id:
            print("Error: --shared-drive-id is required when using --list-files")
            return
        generator.list_files_in_shared_drive(args.folder)
        return

    # For shared drives, require shared drive ID
    if args.shared_drive_id is None:
        print("Warning: No shared drive ID specified. Using personal Google Drive.")
        print("To use a shared drive, first run with --list-drives to find the ID")
        print("Then run with --shared-drive-id <ID>")

    # Run continuous generation
    generator.run_continuous_generation(
        filename=args.filename,
        interval_seconds=args.interval,
        records_per_batch=args.records,
        folder_name=args.folder,
    )


if __name__ == "__main__":
    main()
