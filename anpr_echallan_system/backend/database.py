"""
database.py - MySQL Connection Pool and Batch Insert Queries
Supports production MySQL connection pooling with automatic resilient fallback
to SQLite if MySQL is offline or not configured, ensuring zero downtime.
"""

import os
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("echallan_db")
logger.setLevel(logging.INFO)

# Database Configuration (Environment variables with defaults)
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "echallan_db")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", 5))

# Local SQLite fallback database path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB_PATH = os.path.join(BASE_DIR, "echallan_local.db")

mysql_pool = None
use_mysql = False

def init_db():
    """Initializes the database connection pool and creates required tables."""
    global mysql_pool, use_mysql
    
    # Try initializing MySQL Connection Pool
    try:
        import mysql.connector
        from mysql.connector import pooling
        
        # Test connection or create database if needed
        tmp_conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            port=MYSQL_PORT
        )
        cursor = tmp_conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}")
        cursor.close()
        tmp_conn.close()

        # Initialize Pool
        mysql_pool = pooling.MySQLConnectionPool(
            pool_name="echallan_pool",
            pool_size=POOL_SIZE,
            pool_reset_session=True,
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            port=MYSQL_PORT
        )
        use_mysql = True
        logger.info(f"Connected to MySQL database '{MYSQL_DATABASE}' on {MYSQL_HOST}:{MYSQL_PORT}")
        _create_mysql_tables()
        return
    except Exception as e:
        logger.warning(f"MySQL unavailable ({e}). Falling back to SQLite database at {SQLITE_DB_PATH}")
        use_mysql = False

    # Fallback to SQLite
    _init_sqlite_tables()
    logger.info("Initialized local SQLite database successfully.")

def _create_mysql_tables():
    """Create schema in MySQL."""
    conn = mysql_pool.get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS challans (
                id INT AUTO_INCREMENT PRIMARY KEY,
                challan_no VARCHAR(64) UNIQUE NOT NULL,
                timestamp DATETIME NOT NULL,
                license_plate VARCHAR(32) NOT NULL,
                vehicle_type VARCHAR(32) NOT NULL,
                violation_type VARCHAR(64) NOT NULL,
                fine_amount INT NOT NULL,
                location VARCHAR(128) NOT NULL,
                evidence_frame_path VARCHAR(255),
                plate_crop_path VARCHAR(255),
                confidence FLOAT DEFAULT 0.0,
                status VARCHAR(32) DEFAULT 'PENDING',
                payment_date DATETIME NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_plate (license_plate),
                INDEX idx_timestamp (timestamp),
                INDEX idx_status (status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traffic_stats (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                vehicles_detected INT DEFAULT 0,
                violations_count INT DEFAULT 0
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def _init_sqlite_tables():
    """Create schema in SQLite."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS challans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            challan_no TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            license_plate TEXT NOT NULL,
            vehicle_type TEXT NOT NULL,
            violation_type TEXT NOT NULL,
            fine_amount INTEGER NOT NULL,
            location TEXT NOT NULL,
            evidence_frame_path TEXT,
            plate_crop_path TEXT,
            confidence REAL DEFAULT 0.0,
            status TEXT DEFAULT 'PENDING',
            payment_date TEXT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traffic_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            vehicles_detected INTEGER DEFAULT 0,
            violations_count INTEGER DEFAULT 0
        );
    """)
    conn.commit()
    conn.close()

def add_violation(data: Dict[str, Any]) -> int:
    """Inserts a single violation challan."""
    timestamp_str = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    if use_mysql and mysql_pool:
        conn = mysql_pool.get_connection()
        cursor = conn.cursor()
        try:
            sql = """
                INSERT INTO challans 
                (challan_no, timestamp, license_plate, vehicle_type, violation_type, 
                 fine_amount, location, evidence_frame_path, plate_crop_path, confidence, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            val = (
                data["challan_no"],
                timestamp_str,
                data.get("license_plate", "UNKNOWN"),
                data.get("vehicle_type", "Unknown"),
                data.get("violation_type", "VIOLATION"),
                data.get("fine_amount", 1000),
                data.get("location", "Intersection-01"),
                data.get("evidence_frame_path", ""),
                data.get("plate_crop_path", ""),
                float(data.get("confidence", 0.95)),
                data.get("status", "PENDING")
            )
            cursor.execute(sql, val)
            conn.commit()
            return cursor.lastrowid
        finally:
            cursor.close()
            conn.close()
    else:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        sql = """
            INSERT INTO challans 
            (challan_no, timestamp, license_plate, vehicle_type, violation_type, 
             fine_amount, location, evidence_frame_path, plate_crop_path, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        val = (
            data["challan_no"],
            timestamp_str,
            data.get("license_plate", "UNKNOWN"),
            data.get("vehicle_type", "Unknown"),
            data.get("violation_type", "VIOLATION"),
            data.get("fine_amount", 1000),
            data.get("location", "Intersection-01"),
            data.get("evidence_frame_path", ""),
            data.get("plate_crop_path", ""),
            float(data.get("confidence", 0.95)),
            data.get("status", "PENDING")
        )
        cursor.execute(sql, val)
        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id

def batch_insert_violations(violations: List[Dict[str, Any]]) -> int:
    """Performs an efficient batch insert for multiple violations."""
    if not violations:
        return 0

    if use_mysql and mysql_pool:
        conn = mysql_pool.get_connection()
        cursor = conn.cursor()
        try:
            sql = """
                INSERT INTO challans 
                (challan_no, timestamp, license_plate, vehicle_type, violation_type, 
                 fine_amount, location, evidence_frame_path, plate_crop_path, confidence, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = [
                (
                    v["challan_no"],
                    v.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    v.get("license_plate", "UNKNOWN"),
                    v.get("vehicle_type", "Unknown"),
                    v.get("violation_type", "VIOLATION"),
                    v.get("fine_amount", 1000),
                    v.get("location", "Intersection-01"),
                    v.get("evidence_frame_path", ""),
                    v.get("plate_crop_path", ""),
                    float(v.get("confidence", 0.95)),
                    v.get("status", "PENDING")
                )
                for v in violations
            ]
            cursor.executemany(sql, params)
            conn.commit()
            return cursor.rowcount
        finally:
            cursor.close()
            conn.close()
    else:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        sql = """
            INSERT INTO challans 
            (challan_no, timestamp, license_plate, vehicle_type, violation_type, 
             fine_amount, location, evidence_frame_path, plate_crop_path, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = [
            (
                v["challan_no"],
                v.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                v.get("license_plate", "UNKNOWN"),
                v.get("vehicle_type", "Unknown"),
                v.get("violation_type", "VIOLATION"),
                v.get("fine_amount", 1000),
                v.get("location", "Intersection-01"),
                v.get("evidence_frame_path", ""),
                v.get("plate_crop_path", ""),
                float(v.get("confidence", 0.95)),
                v.get("status", "PENDING")
            )
            for v in violations
        ]
        cursor.executemany(sql, params)
        conn.commit()
        count = cursor.rowcount
        conn.close()
        return count

def get_violations(
    limit: int = 50, 
    offset: int = 0,
    violation_type: Optional[str] = None, 
    plate: Optional[str] = None, 
    status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetches list of violations with filtering and pagination."""
    query = "SELECT id, challan_no, timestamp, license_plate, vehicle_type, violation_type, fine_amount, location, evidence_frame_path, plate_crop_path, confidence, status, payment_date FROM challans WHERE 1=1"
    filters = []
    
    if violation_type and violation_type != "ALL":
        query += " AND violation_type = ?" if not use_mysql else " AND violation_type = %s"
        filters.append(violation_type)
        
    if plate:
        query += " AND license_plate LIKE ?" if not use_mysql else " AND license_plate LIKE %s"
        filters.append(f"%{plate}%")
        
    if status and status != "ALL":
        query += " AND status = ?" if not use_mysql else " AND status = %s"
        filters.append(status)

    query += " ORDER BY id DESC LIMIT ? OFFSET ?" if not use_mysql else " ORDER BY id DESC LIMIT %s OFFSET %s"
    filters.extend([limit, offset])

    results = []
    if use_mysql and mysql_pool:
        conn = mysql_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(query, filters)
            results = cursor.fetchall()
            # Convert datetime to string for JSON serialization
            for r in results:
                if isinstance(r.get("timestamp"), datetime):
                    r["timestamp"] = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(r.get("payment_date"), datetime):
                    r["payment_date"] = r["payment_date"].strftime("%Y-%m-%d %H:%M:%S")
        finally:
            cursor.close()
            conn.close()
    else:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, filters)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        conn.close()
        
    return results

def get_violation_by_id(violation_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve single violation record by ID."""
    query = "SELECT * FROM challans WHERE id = ?" if not use_mysql else "SELECT * FROM challans WHERE id = %s"
    
    if use_mysql and mysql_pool:
        conn = mysql_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(query, (violation_id,))
            row = cursor.fetchone()
            if row:
                if isinstance(row.get("timestamp"), datetime):
                    row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            return row
        finally:
            cursor.close()
            conn.close()
    else:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, (violation_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

def update_violation_status(violation_id: int, new_status: str) -> bool:
    """Updates the status of a challan (e.g. PAID, DISPUTED, CANCELLED)."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if new_status == "PAID" else None
    
    if use_mysql and mysql_pool:
        conn = mysql_pool.get_connection()
        cursor = conn.cursor()
        try:
            sql = "UPDATE challans SET status = %s, payment_date = %s WHERE id = %s"
            cursor.execute(sql, (new_status, now_str, violation_id))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()
            conn.close()
    else:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        sql = "UPDATE challans SET status = ?, payment_date = ? WHERE id = ?"
        cursor.execute(sql, (new_status, now_str, violation_id))
        conn.commit()
        updated = cursor.rowcount > 0
        conn.close()
        return updated

def get_dashboard_stats() -> Dict[str, Any]:
    """Retrieves aggregated statistics for the command center dashboard."""
    stats = {
        "total_violations": 0,
        "total_fines": 0,
        "pending_challans": 0,
        "paid_challans": 0,
        "violation_types": {},
        "recent_trends": []
    }
    
    if use_mysql and mysql_pool:
        conn = mysql_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT COUNT(*) as total, COALESCE(SUM(fine_amount), 0) as total_fines FROM challans")
            summary = cursor.fetchone()
            if summary:
                stats["total_violations"] = summary["total"]
                stats["total_fines"] = int(summary["total_fines"])
                
            cursor.execute("SELECT status, COUNT(*) as cnt FROM challans GROUP BY status")
            for r in cursor.fetchall():
                if r["status"] == "PAID":
                    stats["paid_challans"] = r["cnt"]
                else:
                    stats["pending_challans"] += r["cnt"]
                    
            cursor.execute("SELECT violation_type, COUNT(*) as cnt FROM challans GROUP BY violation_type")
            for r in cursor.fetchall():
                stats["violation_types"][r["violation_type"]] = r["cnt"]
        finally:
            cursor.close()
            conn.close()
    else:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total, COALESCE(SUM(fine_amount), 0) as total_fines FROM challans")
        summary = cursor.fetchone()
        if summary:
            stats["total_violations"] = summary["total"]
            stats["total_fines"] = int(summary["total_fines"])

        cursor.execute("SELECT status, COUNT(*) as cnt FROM challans GROUP BY status")
        for r in cursor.fetchall():
            if r["status"] == "PAID":
                stats["paid_challans"] = r["cnt"]
            else:
                stats["pending_challans"] += r["cnt"]

        cursor.execute("SELECT violation_type, COUNT(*) as cnt FROM challans GROUP BY violation_type")
        for r in cursor.fetchall():
            stats["violation_types"][r["violation_type"]] = r["cnt"]
        conn.close()

    return stats

