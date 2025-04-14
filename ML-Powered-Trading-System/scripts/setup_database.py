#!/usr/bin/env python
"""
setup_database.py - Database Setup and Migration Script

This script initializes and migrates the database schema for the ML-powered trading system.
It creates necessary tables, indexes, and functions for optimal performance.

Usage:
    python -m scripts.setup_database [options]

Options:
    --config CONFIG         Path to database configuration file
    --drop                  Drop existing tables before creation (use with caution)
    --seed                  Seed the database with initial data
    --migrate               Run only pending migrations
    --dry-run               Show SQL statements without executing
    --backup                Create database backup before changes
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import tempfile

# Add project root to path
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(project_root, "logs", "database_setup.log"))
    ]
)

logger = logging.getLogger("database_setup")


class DatabaseSetup:
    """
    Database setup and migration manager for the ML-powered trading system.
    Handles schema creation, migrations, and data seeding.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the database setup.
        
        Args:
            config_path: Path to database configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.db_type = self.config.get("type", "postgresql").lower()
        self.migrations_dir = Path(project_root, "data", "migrations")
        self.schema_dir = Path(project_root, "data", "schema")
        self.seed_dir = Path(project_root, "data", "seed")
        
        # Create directories if they don't exist
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        self.seed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Database setup initialized for {self.db_type} database")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load database configuration.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _get_connection_string(self) -> str:
        """
        Get database connection string.
        
        Returns:
            Connection string
        """
        db_config = self.config
        
        if self.db_type == "postgresql":
            host = db_config.get("host", "localhost")
            port = db_config.get("port", 5432)
            dbname = db_config.get("database", "trading_system")
            user = db_config.get("user", "postgres")
            password = db_config.get("password", "")
            
            return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
            
        elif self.db_type == "mysql":
            host = db_config.get("host", "localhost")
            port = db_config.get("port", 3306)
            dbname = db_config.get("database", "trading_system")
            user = db_config.get("user", "root")
            password = db_config.get("password", "")
            
            return f"mysql://{user}:{password}@{host}:{port}/{dbname}"
            
        elif self.db_type == "sqlite":
            dbpath = db_config.get("path", "data/database/trading_system.db")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(dbpath), exist_ok=True)
            
            return f"sqlite:///{dbpath}"
            
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def _execute_sql(self, sql: str, dry_run: bool = False) -> bool:
        """
        Execute SQL statement on the database.
        
        Args:
            sql: SQL statement to execute
            dry_run: If True, print SQL without executing
            
        Returns:
            True if successful, False otherwise
        """
        if dry_run:
            print(f"\n--- SQL Statement ---\n{sql}\n--------------------")
            return True
        
        conn_string = self._get_connection_string()
        
        if self.db_type == "postgresql":
            # Use psql command line tool
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.sql') as tmp:
                    tmp.write(sql)
                    tmp.flush()
                    
                    # Extract connection parameters
                    parts = conn_string.replace("postgresql://", "").split("/")
                    auth_host = parts[0].split("@")
                    user_pass = auth_host[0].split(":")
                    host_port = auth_host[1].split(":")
                    
                    user = user_pass[0]
                    password = user_pass[1] if len(user_pass) > 1 else ""
                    host = host_port[0]
                    port = host_port[1] if len(host_port) > 1 else "5432"
                    dbname = parts[1]
                    
                    # Set password environment variable
                    env = os.environ.copy()
                    if password:
                        env["PGPASSWORD"] = password
                    
                    # Build psql command
                    cmd = [
                        "psql",
                        "-h", host,
                        "-p", port,
                        "-U", user,
                        "-d", dbname,
                        "-f", tmp.name,
                        "-v", "ON_ERROR_STOP=1"
                    ]
                    
                    # Execute command
                    result = subprocess.run(
                        cmd,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        logger.error(f"SQL execution failed: {result.stderr}")
                        return False
                    
                    logger.debug(f"SQL execution successful: {result.stdout}")
                    return True
                    
            except Exception as e:
                logger.error(f"Error executing SQL: {str(e)}")
                return False
                
        elif self.db_type == "mysql":
            # Use mysql command line tool
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.sql') as tmp:
                    tmp.write(sql)
                    tmp.flush()
                    
                    # Extract connection parameters
                    parts = conn_string.replace("mysql://", "").split("/")
                    auth_host = parts[0].split("@")
                    user_pass = auth_host[0].split(":")
                    host_port = auth_host[1].split(":")
                    
                    user = user_pass[0]
                    password = user_pass[1] if len(user_pass) > 1 else ""
                    host = host_port[0]
                    port = host_port[1] if len(host_port) > 1 else "3306"
                    dbname = parts[1]
                    
                    # Build mysql command
                    cmd = [
                        "mysql",
                        "-h", host,
                        "-P", port,
                        "-u", user
                    ]
                    
                    if password:
                        cmd.extend(["-p" + password])
                    
                    cmd.extend([
                        dbname,
                        "-e", f"source {tmp.name}"
                    ])
                    
                    # Execute command
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        logger.error(f"SQL execution failed: {result.stderr}")
                        return False
                    
                    logger.debug(f"SQL execution successful: {result.stdout}")
                    return True
                    
            except Exception as e:
                logger.error(f"Error executing SQL: {str(e)}")
                return False
                
        elif self.db_type == "sqlite":
            # Use sqlite3 command line tool
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.sql') as tmp:
                    tmp.write(sql)
                    tmp.flush()
                    
                    # Extract database path
                    dbpath = conn_string.replace("sqlite:///", "")
                    
                    # Build sqlite3 command
                    cmd = [
                        "sqlite3",
                        dbpath,
                        ".read " + tmp.name
                    ]
                    
                    # Execute command
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        logger.error(f"SQL execution failed: {result.stderr}")
                        return False
                    
                    logger.debug(f"SQL execution successful: {result.stdout}")
                    return True
                    
            except Exception as e:
                logger.error(f"Error executing SQL: {str(e)}")
                return False
        
        else:
            logger.error(f"Unsupported database type: {self.db_type}")
            return False
    
    def create_database(self, drop_existing: bool = False, dry_run: bool = False) -> bool:
        """
        Create the database schema.
        
        Args:
            drop_existing: If True, drop existing tables before creation
            dry_run: If True, print SQL without executing
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Creating database schema")
        
        # Load schema SQL
        schema_file = self.schema_dir / f"{self.db_type}_schema.sql"
        
        if not schema_file.exists():
            logger.error(f"Schema file not found: {schema_file}")
            return False
        
        try:
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            # Add drop tables if requested
            if drop_existing:
                logger.warning("Dropping existing tables")
                drop_sql = self._get_drop_tables_sql()
                schema_sql = drop_sql + "\n\n" + schema_sql
            
            # Execute schema SQL
            if not self._execute_sql(schema_sql, dry_run):
                logger.error("Failed to create database schema")
                return False
            
            logger.info("Database schema created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating database schema: {str(e)}")
            return False
    
    def _get_drop_tables_sql(self) -> str:
        """
        Get SQL to drop all tables.
        
        Returns:
            SQL for dropping tables
        """
        if self.db_type == "postgresql":
            return """
DO $$ 
DECLARE
    r RECORD;
BEGIN
    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = current_schema()) 
    LOOP
        EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
    END LOOP;
END $$;
"""
        elif self.db_type == "mysql":
            return """
SET FOREIGN_KEY_CHECKS = 0;
SET GROUP_CONCAT_MAX_LEN=32768;
SET @tables = NULL;
SELECT GROUP_CONCAT('`', table_schema, '`.`', table_name, '`') INTO @tables
  FROM information_schema.tables 
  WHERE table_schema = DATABASE();
SELECT IFNULL(@tables,'dummy') INTO @tables;
SET @tables = CONCAT('DROP TABLE IF EXISTS ', @tables);
PREPARE stmt FROM @tables;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;
SET FOREIGN_KEY_CHECKS = 1;
"""
        elif self.db_type == "sqlite":
            return """
PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;
SELECT 'DROP TABLE IF EXISTS ' || name || ';' FROM sqlite_master WHERE type = 'table';
COMMIT;
PRAGMA foreign_keys = ON;
"""
        else:
            return "-- Unsupported database type for drop tables"
    
    def run_migrations(self, dry_run: bool = False) -> bool:
        """
        Run pending database migrations.
        
        Args:
            dry_run: If True, print SQL without executing
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Running database migrations")
        
        # Check if migrations table exists
        migrations_table_sql = self._get_migrations_table_sql()
        if not self._execute_sql(migrations_table_sql, dry_run):
            logger.error("Failed to create migrations table")
            return False
        
        # Get list of applied migrations
        applied_migrations = self._get_applied_migrations(dry_run)
        if applied_migrations is None:
            logger.error("Failed to get applied migrations")
            return False
        
        # Get list of available migrations
        available_migrations = []
        for file in sorted(self.migrations_dir.glob(f"{self.db_type}_*.sql")):
            migration_id = file.stem.replace(f"{self.db_type}_", "")
            available_migrations.append({
                "id": migration_id,
                "file": file
            })
        
        logger.info(f"Found {len(available_migrations)} available migrations")
        
        # Run pending migrations
        pending_migrations = [m for m in available_migrations if m["id"] not in applied_migrations]
        
        if not pending_migrations:
            logger.info("No pending migrations to apply")
            return True
        
        logger.info(f"Applying {len(pending_migrations)} pending migrations")
        
        for migration in pending_migrations:
            migration_id = migration["id"]
            migration_file = migration["file"]
            
            logger.info(f"Applying migration: {migration_id}")
            
            try:
                with open(migration_file, 'r') as f:
                    migration_sql = f.read()
                
                # Execute migration SQL
                if not self._execute_sql(migration_sql, dry_run):
                    logger.error(f"Failed to apply migration: {migration_id}")
                    return False
                
                # Record migration
                if not dry_run:
                    record_sql = self._get_record_migration_sql(migration_id)
                    if not self._execute_sql(record_sql, dry_run):
                        logger.error(f"Failed to record migration: {migration_id}")
                        return False
                
                logger.info(f"Migration {migration_id} applied successfully")
                
            except Exception as e:
                logger.error(f"Error applying migration {migration_id}: {str(e)}")
                return False
        
        logger.info("All migrations applied successfully")
        return True
    
    def _get_migrations_table_sql(self) -> str:
        """
        Get SQL to create migrations table.
        
        Returns:
            SQL for creating migrations table
        """
        if self.db_type == "postgresql":
            return """
CREATE TABLE IF NOT EXISTS migrations (
    id VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""
        elif self.db_type == "mysql":
            return """
CREATE TABLE IF NOT EXISTS migrations (
    id VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""
        elif self.db_type == "sqlite":
            return """
CREATE TABLE IF NOT EXISTS migrations (
    id VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""
        else:
            return "-- Unsupported database type for migrations table"
    
    def _get_record_migration_sql(self, migration_id: str) -> str:
        """
        Get SQL to record a migration.
        
        Args:
            migration_id: Migration ID
            
        Returns:
            SQL for recording migration
        """
        return f"""
INSERT INTO migrations (id) VALUES ('{migration_id}');
"""
    
    def _get_applied_migrations(self, dry_run: bool = False) -> Optional[List[str]]:
        """
        Get list of applied migrations.
        
        Args:
            dry_run: If True, return empty list
            
        Returns:
            List of applied migration IDs
        """
        if dry_run:
            return []
        
        try:
            # Execute query to get applied migrations
            query = "SELECT id FROM migrations ORDER BY applied_at;"
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.sql') as tmp:
                # Write query to temp file
                tmp.write(query)
                tmp.flush()
                
                # Execute query
                conn_string = self._get_connection_string()
                
                if self.db_type == "postgresql":
                    # Extract connection parameters
                    parts = conn_string.replace("postgresql://", "").split("/")
                    auth_host = parts[0].split("@")
                    user_pass = auth_host[0].split(":")
                    host_port = auth_host[1].split(":")
                    
                    user = user_pass[0]
                    password = user_pass[1] if len(user_pass) > 1 else ""
                    host = host_port[0]
                    port = host_port[1] if len(host_port) > 1 else "5432"
                    dbname = parts[1]
                    
                    # Set password environment variable
                    env = os.environ.copy()
                    if password:
                        env["PGPASSWORD"] = password
                    
                    # Build psql command
                    cmd = [
                        "psql",
                        "-h", host,
                        "-p", port,
                        "-U", user,
                        "-d", dbname,
                        "-t",  # Tuple only output
                        "-c", query
                    ]
                    
                elif self.db_type == "mysql":
                    # Extract connection parameters
                    parts = conn_string.replace("mysql://", "").split("/")
                    auth_host = parts[0].split("@")
                    user_pass = auth_host[0].split(":")
                    host_port = auth_host[1].split(":")
                    
                    user = user_pass[0]
                    password = user_pass[1] if len(user_pass) > 1 else ""
                    host = host_port[0]
                    port = host_port[1] if len(host_port) > 1 else "3306"
                    dbname = parts[1]
                    
                    # Build mysql command
                    cmd = [
                        "mysql",
                        "-h", host,
                        "-P", port,
                        "-u", user,
                        "-s",  # Silent mode
                        "-N",  # Skip column names
                    ]
                    
                    if password:
                        cmd.extend(["-p" + password])
                    
                    cmd.extend([
                        dbname,
                        "-e", query
                    ])
                    
                elif self.db_type == "sqlite":
                    # Extract database path
                    dbpath = conn_string.replace("sqlite:///", "")
                    
                    # Build sqlite3 command
                    cmd = [
                        "sqlite3",
                        dbpath,
                        "-csv",
                        query
                    ]
                
                else:
                    logger.error(f"Unsupported database type: {self.db_type}")
                    return None
                
                # Execute command
                try:
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        if "relation \"migrations\" does not exist" in result.stderr:
                            # Migrations table doesn't exist yet
                            return []
                        else:
                            logger.error(f"Error getting applied migrations: {result.stderr}")
                            return None
                    
                    # Parse output
                    migrations = []
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            migrations.append(line.strip())
                    
                    return migrations
                    
                except Exception as e:
                    logger.error(f"Error executing query: {str(e)}")
                    return None
            
        except Exception as e:
            logger.error(f"Error getting applied migrations: {str(e)}")
            return None
    
    def seed_database(self, dry_run: bool = False) -> bool:
        """
        Seed the database with initial data.
        
        Args:
            dry_run: If True, print SQL without executing
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Seeding database with initial data")
        
        # Load seed SQL
        seed_file = self.seed_dir / f"{self.db_type}_seed.sql"
        
        if not seed_file.exists():
            logger.error(f"Seed file not found: {seed_file}")
            return False
        
        try:
            with open(seed_file, 'r') as f:
                seed_sql = f.read()
            
            # Execute seed SQL
            if not self._execute_sql(seed_sql, dry_run):
                logger.error("Failed to seed database")
                return False
            
            logger.info("Database seeded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error seeding database: {str(e)}")
            return False
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path to store backup file
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Creating database backup")
        
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backups/database_backup_{timestamp}.sql"
        
        # Ensure backup directory exists
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        conn_string = self._get_connection_string()
        
        if self.db_type == "postgresql":
            try:
                # Extract connection parameters
                parts = conn_string.replace("postgresql://", "").split("/")
                auth_host = parts[0].split("@")
                user_pass = auth_host[0].split(":")
                host_port = auth_host[1].split(":")
                
                user = user_pass[0]
                password = user_pass[1] if len(user_pass) > 1 else ""
                host = host_port[0]
                port = host_port[1] if len(host_port) > 1 else "5432"
                dbname = parts[1]
                
                # Set password environment variable
                env = os.environ.copy()
                if password:
                    env["PGPASSWORD"] = password
                
                # Build pg_dump command
                cmd = [
                    "pg_dump",
                    "-h", host,
                    "-p", port,
                    "-U", user,
                    "-d", dbname,
                    "-f", backup_path,
                    "--clean"
                ]
                
                # Execute command
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"Backup failed: {result.stderr}")
                    return False
                
                logger.info(f"Database backup created at {backup_path}")
                return True
                
            except Exception as e:
                logger.error(f"Error creating backup: {str(e)}")
                return False
                
        elif self.db_type == "sqlite":
            try:
                # Extract database path
                dbpath = conn_string.replace("sqlite:///", "")
                
                # Build sqlite3 command
                cmd = [
                    "sqlite3",
                    dbpath,
                    ".dump"
                ]
                
                # Execute command and save output to file
                with open(backup_path, 'w') as f:
                    result = subprocess.run(
                        cmd,
                        stdout=f,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                
                if result.returncode != 0:
                    logger.error(f"Backup failed: {result.stderr}")
                    return False
                
                logger.info(f"Database backup created at {backup_path}")
                return True
                
            except Exception as e:
                logger.error(f"Error creating backup: {str(e)}")
                return False
        
        else:
            logger.error(f"Unsupported database type: {self.db_type}")
            return False


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Setup database schema and migrations')
    parser.add_argument('--config', type=str, default='config/database_config.json',
                      help='Path to database configuration file')
    parser.add_argument('--drop', action='store_true',
                      help='Drop existing tables before creation')
    parser.add_argument('--seed', action='store_true',
                      help='Seed database with initial data')
    parser.add_argument('--migrate', action='store_true',
                      help='Run only pending migrations')
    parser.add_argument('--dry-run', action='store_true',
                      help='Show SQL statements without executing')
    parser.add_argument('--backup', action='store_true',
                      help='Create database backup before changes')
    
    args = parser.parse_args()
    
    try:
        # Initialize database setup
        db_setup = DatabaseSetup(args.config)
        
        # Create backup if requested
        if args.backup:
            logger.info("Creating database backup")
            if not db_setup.backup_database():
                logger.error("Database backup failed")
                sys.exit(1)
        
        # Run migrations only if requested
        if args.migrate:
            logger.info("Running pending migrations")
            if not db_setup.run_migrations(args.dry_run):
                logger.error("Database migration failed")
                sys.exit(1)
        else:
            # Create database schema
            logger.info("Creating database schema")
            if not db_setup.create_database(args.drop, args.dry_run):
                logger.error("Database schema creation failed")
                sys.exit(1)
            
            # Run migrations
            logger.info("Running pending migrations")
            if not db_setup.run_migrations(args.dry_run):
                logger.error("Database migration failed")
                sys.exit(1)
        
        # Seed database if requested
        if args.seed:
            logger.info("Seeding database")
            if not db_setup.seed_database(args.dry_run):
                logger.error("Database seeding failed")
                sys.exit(1)
        
        logger.info("Database setup completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"Backup failed: {result.stderr}")
                    return False
                
                logger.info(f"Database backup created at {backup_path}")
                return True
                
            except Exception as e:
                logger.error(f"Error creating backup: {str(e)}")
                return False
                
        elif self.db_type == "mysql":
            try:
                # Extract connection parameters
                parts = conn_string.replace("mysql://", "").split("/")
                auth_host = parts[0].split("@")
                user_pass = auth_host[0].split(":")
                host_port = auth_host[1].split(":")
                
                user = user_pass[0]
                password = user_pass[1] if len(user_pass) > 1 else ""
                host = host_port[0]
                port = host_port[1] if len(host_port) > 1 else "3306"
                dbname = parts[1]
                
                # Build mysqldump command
                cmd = [
                    "mysqldump",
                    "-h", host,
                    "-P", port,
                    "-u", user,
                ]
                
                if password:
                    cmd.extend([f"-p{password}"])
                
                cmd.extend([
                    "--databases", dbname,
                    "--routines",
                    "--triggers",
                    "--add-drop-table",
                    f"--result-file={backup_path}"
                ])
                
                # Execute command
                result = subprocess.run(