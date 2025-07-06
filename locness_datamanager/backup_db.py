#!/usr/bin/env python3
import sqlite3
import shutil
import os
from datetime import datetime, timedelta
from pathlib import Path

class DatabaseBackup:
    def __init__(self, db_path, backup_dir="backups"):
        self.db_path = db_path
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self):
        """Create timestamped backup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"sensors_{timestamp}.db"
        
        source = sqlite3.connect(self.db_path)
        backup = sqlite3.connect(str(backup_path))
        
        try:
            source.backup(backup)
            backup.close()
            source.close()
            print(f"Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"Backup failed: {e}")
            return None
    
    def cleanup_old_backups(self, keep_days=7):
        """Remove backups older than keep_days"""
        cutoff = datetime.now() - timedelta(days=keep_days)
        
        for backup_file in self.backup_dir.glob("sensors_*.db"):
            if backup_file.stat().st_mtime < cutoff.timestamp():
                backup_file.unlink()
                print(f"Removed old backup: {backup_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backup SQLite DB and clean up old backups.")
    parser.add_argument('--db', type=str, default="sensors.db", help="Path to SQLite database (default: sensors.db)")
    parser.add_argument('--backup-dir', type=str, default="backups", help="Directory to store backups (default: backups)")
    parser.add_argument('--keep-days', type=int, default=7, help="Days to keep backups (default: 7)")
    args = parser.parse_args()

    backup_manager = DatabaseBackup(args.db, backup_dir=args.backup_dir)
    backup_manager.create_backup()
    backup_manager.cleanup_old_backups(keep_days=args.keep_days)

# Usage
if __name__ == "__main__":
    main()