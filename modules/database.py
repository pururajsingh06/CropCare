"""
CropCare AI - Database Module
Manages SQLite storage for crop disease detection history,
providing search, filtering, analytical aggregations, and CSV export.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import sqlite3
import json
import pandas as pd
from config import DATABASE_PATH


def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite database with row dict factory."""
    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the detections table if it does not already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT,
            crop_name TEXT NOT NULL,
            disease_name TEXT NOT NULL,
            raw_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            health_status TEXT NOT NULL,
            top_predictions_json TEXT,
            model_used TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_detection(
    crop_name: str,
    disease_name: str,
    raw_class: str,
    confidence: float,
    health_status: str,
    image_path: Optional[str] = None,
    top_predictions: Optional[List[Dict[str, Any]]] = None,
    model_used: Optional[str] = "MobileNetV2"
) -> int:
    """
    Saves a successful disease prediction record to the database.
    Returns the newly inserted record ID.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_json = json.dumps(top_predictions) if top_predictions else None

    cursor.execute("""
        INSERT INTO detections (
            timestamp, image_path, crop_name, disease_name,
            raw_class, confidence, health_status, top_predictions_json, model_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        image_path,
        crop_name,
        disease_name,
        raw_class,
        float(confidence),
        health_status,
        top_json,
        model_used
    ))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_all_detections(
    search: Optional[str] = None,
    crop_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves filtered detection records ordered newest first.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM detections WHERE 1=1"
    params = []

    if search:
        search_pattern = f"%{search.strip()}%"
        query += " AND (crop_name LIKE ? OR disease_name LIKE ? OR raw_class LIKE ?)"
        params.extend([search_pattern, search_pattern, search_pattern])

    if crop_filter and crop_filter != "All Crops":
        query += " AND crop_name = ?"
        params.append(crop_filter)

    if status_filter and status_filter != "All":
        query += " AND health_status = ?"
        params.append(status_filter)

    if date_from:
        query += " AND date(timestamp) >= date(?)"
        params.append(str(date_from))

    if date_to:
        query += " AND date(timestamp) <= date(?)"
        params.append(str(date_to))

    query += " ORDER BY id DESC"

    if limit:
        query += f" LIMIT {int(limit)}"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results


def get_detection_by_id(detection_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single detection by its primary key ID."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections WHERE id = ?", (int(detection_id),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_detection(detection_id: int) -> bool:
    """Deletes a single detection record by ID."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections WHERE id = ?", (int(detection_id),))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def clear_all_detections() -> int:
    """Clears all records from detections table. Returns count deleted."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected


def get_dashboard_statistics() -> Dict[str, Any]:
    """
    Computes aggregate metrics from actual SQLite database records for dashboard & analytics.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM detections")
    total_scans = cursor.fetchone()[0]

    if total_scans == 0:
        conn.close()
        return {
            "total_scans": 0,
            "healthy_count": 0,
            "diseased_count": 0,
            "healthy_pct": 0.0,
            "diseased_pct": 0.0,
            "avg_confidence": 0.0,
            "crops_scanned_count": 0,
            "crop_distribution": {},
            "disease_distribution": {},
            "recent_records": []
        }

    cursor.execute("SELECT COUNT(*) FROM detections WHERE health_status = 'Healthy'")
    healthy_count = cursor.fetchone()[0]
    diseased_count = total_scans - healthy_count

    cursor.execute("SELECT AVG(confidence) FROM detections")
    avg_conf = cursor.fetchone()[0] or 0.0

    cursor.execute("SELECT crop_name, COUNT(*) as cnt FROM detections GROUP BY crop_name ORDER BY cnt DESC")
    crop_dist = {row["crop_name"]: row["cnt"] for row in cursor.fetchall()}

    cursor.execute("SELECT disease_name, COUNT(*) as cnt FROM detections WHERE health_status != 'Healthy' GROUP BY disease_name ORDER BY cnt DESC")
    disease_dist = {row["disease_name"]: row["cnt"] for row in cursor.fetchall()}

    cursor.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 5")
    recent_records = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return {
        "total_scans": total_scans,
        "healthy_count": healthy_count,
        "diseased_count": diseased_count,
        "healthy_pct": round((healthy_count / total_scans) * 100.0, 1),
        "diseased_pct": round((diseased_count / total_scans) * 100.0, 1),
        "avg_confidence": round(avg_conf, 2),
        "crops_scanned_count": len(crop_dist),
        "crop_distribution": crop_dist,
        "disease_distribution": disease_dist,
        "recent_records": recent_records
    }


def get_detections_dataframe() -> pd.DataFrame:
    """Returns all detections as a pandas DataFrame."""
    init_db()
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM detections ORDER BY id DESC", conn)
    conn.close()
    return df


def export_history_csv() -> bytes:
    """Exports all detection records as CSV bytes ready for download."""
    df = get_detections_dataframe()
    if df.empty:
        # Return empty CSV with standard headers
        df = pd.DataFrame(columns=[
            "id", "timestamp", "crop_name", "disease_name",
            "confidence", "health_status", "model_used", "image_path"
        ])
    return df.to_csv(index=False).encode("utf-8")
