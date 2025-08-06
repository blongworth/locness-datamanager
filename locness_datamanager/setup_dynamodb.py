#!/usr/bin/env python3
"""
setup_dynamodb.py

Script to create a DynamoDB table on AWS for storing underway summary data.
The table is designed to store oceanographic data with timestamp-based partitioning.
"""

import argparse
import boto3
import logging
import sys
from botocore.exceptions import ClientError


def create_underway_summary_table(
    table_name: str,
    region_name: str = 'us-east-1',
    billing_mode: str = 'PAY_PER_REQUEST',
    read_capacity: int = 5,
    write_capacity: int = 5
) -> bool:
    """
    Create a DynamoDB table for underway summary data.
    
    Args:
        table_name: Name of the DynamoDB table to create
        region_name: AWS region where the table will be created
        billing_mode: Either 'PAY_PER_REQUEST' or 'PROVISIONED'
        read_capacity: Read capacity units (only used for PROVISIONED billing)
        write_capacity: Write capacity units (only used for PROVISIONED billing)
        
    Returns:
        True if table was created successfully, False otherwise
    """
    
    try:
        # Initialize DynamoDB client
        dynamodb = boto3.resource('dynamodb', region_name=region_name)
        
        # Check if table already exists
        try:
            existing_table = dynamodb.Table(table_name)
            existing_table.load()
            logging.info(f"Table {table_name} already exists in region {region_name}")
            
            # Print table status
            print(f"\nTable Status: {existing_table.table_status}")
            print(f"Item Count: {existing_table.item_count}")
            print(f"Table Size: {existing_table.table_size_bytes} bytes")
            print(f"Creation Time: {existing_table.creation_date_time}")
            
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceNotFoundException':
                logging.error(f"Error checking table existence: {e}")
                return False
        
        # Define table schema
        key_schema = [
            {
                'AttributeName': 'datetime_utc',
                'KeyType': 'HASH'  # Partition key
            }
        ]
        
        attribute_definitions = [
            {
                'AttributeName': 'datetime_utc',
                'AttributeType': 'S'  # String (ISO format timestamp)
            }
        ]
        
        # Create table parameters
        table_params = {
            'TableName': table_name,
            'KeySchema': key_schema,
            'AttributeDefinitions': attribute_definitions,
        }
        
        # Set billing mode
        if billing_mode.upper() == 'PROVISIONED':
            table_params['BillingMode'] = 'PROVISIONED'
            table_params['ProvisionedThroughput'] = {
                'ReadCapacityUnits': read_capacity,
                'WriteCapacityUnits': write_capacity
            }
        else:
            table_params['BillingMode'] = 'PAY_PER_REQUEST'
        
        # Add tags
        table_params['Tags'] = [
            {'Key': 'Project', 'Value': 'LOCNESS'},
            {'Key': 'DataType', 'Value': 'UnderwayOceanographic'},
            {'Key': 'Environment', 'Value': 'Production'}
        ]
        
        logging.info(f"Creating DynamoDB table {table_name} in region {region_name}...")
        print(f"Creating table: {table_name}")
        print(f"Region: {region_name}")
        print(f"Billing mode: {billing_mode}")
        
        # Create the table
        table = dynamodb.create_table(**table_params)
        
        # Wait for table to be created
        print("Waiting for table to be created...")
        table.wait_until_exists()
        
        # Verify table creation
        table.reload()
        
        logging.info(f"Successfully created table {table_name}")
        print(f"\nTable {table_name} created successfully!")
        print(f"Table Status: {table.table_status}")
        print(f"Table ARN: {table.table_arn}")
        print(f"Creation Time: {table.creation_date_time}")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ResourceInUseException':
            logging.warning(f"Table {table_name} already exists")
            print(f"Table {table_name} already exists")
            return True
        else:
            logging.error(f"Failed to create table {table_name}: {e}")
            print(f"Error creating table: {e}")
            return False
            
    except Exception as e:
        logging.error(f"Unexpected error creating table {table_name}: {e}")
        print(f"Unexpected error: {e}")
        return False


def delete_underway_summary_table(table_name: str, region_name: str = 'us-east-1') -> bool:
    """
    Delete a DynamoDB table.
    
    Args:
        table_name: Name of the DynamoDB table to delete
        region_name: AWS region where the table exists
        
    Returns:
        True if table was deleted successfully, False otherwise
    """
    try:
        dynamodb = boto3.resource('dynamodb', region_name=region_name)
        table = dynamodb.Table(table_name)
        
        # Check if table exists
        try:
            table.load()
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logging.info(f"Table {table_name} does not exist")
                print(f"Table {table_name} does not exist")
                return True
            else:
                raise
        
        # Confirm deletion
        print(f"\nWARNING: This will permanently delete the table '{table_name}' and ALL its data!")
        confirm = input("Type 'DELETE' to confirm: ")
        
        if confirm != 'DELETE':
            print("Deletion cancelled")
            return False
        
        logging.info(f"Deleting table {table_name}...")
        print(f"Deleting table {table_name}...")
        
        table.delete()
        
        # Wait for table to be deleted
        print("Waiting for table to be deleted...")
        table.wait_until_not_exists()
        
        logging.info(f"Successfully deleted table {table_name}")
        print(f"Table {table_name} deleted successfully!")
        
        return True
        
    except ClientError as e:
        logging.error(f"Failed to delete table {table_name}: {e}")
        print(f"Error deleting table: {e}")
        return False
        
    except Exception as e:
        logging.error(f"Unexpected error deleting table {table_name}: {e}")
        print(f"Unexpected error: {e}")
        return False


def list_dynamodb_tables(region_name: str = 'us-east-1'):
    """List all DynamoDB tables in the specified region."""
    try:
        dynamodb = boto3.client('dynamodb', region_name=region_name)
        
        print(f"\nDynamoDB tables in region {region_name}:")
        print("-" * 50)
        
        paginator = dynamodb.get_paginator('list_tables')
        
        table_count = 0
        for page in paginator.paginate():
            for table_name in page['TableNames']:
                table_count += 1
                print(f"{table_count:2d}. {table_name}")
        
        if table_count == 0:
            print("No tables found")
            
    except Exception as e:
        logging.error(f"Error listing tables: {e}")
        print(f"Error listing tables: {e}")


def main():
    """Command-line interface for DynamoDB table management."""
    parser = argparse.ArgumentParser(
        description="Create and manage DynamoDB tables for LOCNESS underway summary data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a table with default settings
  python setup_dynamodb.py create --table-name locness-underway-summary

  # Create a table with provisioned billing
  python setup_dynamodb.py create --table-name locness-underway-summary \\
    --billing-mode PROVISIONED --read-capacity 10 --write-capacity 10

  # List all tables
  python setup_dynamodb.py list

  # Delete a table
  python setup_dynamodb.py delete --table-name locness-underway-summary

Note: Ensure AWS credentials are configured via AWS CLI, environment variables,
or IAM roles before running this script.
        """
    )
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new DynamoDB table')
    create_parser.add_argument('--table-name', '-t', required=True, 
                              help='Name of the DynamoDB table to create')
    create_parser.add_argument('--region', '-r', default='us-east-1',
                              help='AWS region (default: us-east-1)')
    create_parser.add_argument('--billing-mode', choices=['PAY_PER_REQUEST', 'PROVISIONED'],
                              default='PAY_PER_REQUEST', help='Billing mode (default: PAY_PER_REQUEST)')
    create_parser.add_argument('--read-capacity', type=int, default=5,
                              help='Read capacity units for PROVISIONED billing (default: 5)')
    create_parser.add_argument('--write-capacity', type=int, default=5,
                              help='Write capacity units for PROVISIONED billing (default: 5)')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a DynamoDB table')
    delete_parser.add_argument('--table-name', '-t', required=True,
                              help='Name of the DynamoDB table to delete')
    delete_parser.add_argument('--region', '-r', default='us-east-1',
                              help='AWS region (default: us-east-1)')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all DynamoDB tables')
    list_parser.add_argument('--region', '-r', default='us-east-1',
                            help='AWS region (default: us-east-1)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Check AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        logging.info(f"Using AWS account: {identity['Account']}")
        print(f"AWS Account: {identity['Account']}")
        print(f"AWS User/Role: {identity['Arn']}")
    except Exception as e:
        logging.error(f"AWS credentials not configured properly: {e}")
        print(f"Error: AWS credentials not configured properly: {e}")
        print("\nPlease configure AWS credentials using:")
        print("  1. AWS CLI: aws configure")
        print("  2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        print("  3. IAM roles (for EC2 instances)")
        sys.exit(1)
    
    success = True
    
    if args.command == 'create':
        success = create_underway_summary_table(
            table_name=args.table_name,
            region_name=args.region,
            billing_mode=args.billing_mode,
            read_capacity=args.read_capacity,
            write_capacity=args.write_capacity
        )
        
    elif args.command == 'delete':
        success = delete_underway_summary_table(
            table_name=args.table_name,
            region_name=args.region
        )
        
    elif args.command == 'list':
        list_dynamodb_tables(region_name=args.region)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
