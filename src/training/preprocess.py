"""
preprocess.py — Extract Wi-Fi fingerprints from SQLite and prepare for model training.

Usage:
    from preprocess import load_and_preprocess
    X_train, X_test, y_train, y_test, feature_names, label_encoder = load_and_preprocess("path/to/wifi_scans.db")
"""

import sqlite3
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'data', 'raw', 'wifi_scans.db'
)

def load_fingerprints_from_db(db_path=DEFAULT_DB_PATH):
    """
    Query SQLite database and build a fingerprint matrix.
    
    Each row = one scan (identified by scan_id)
    Each column = one unique BSSID (access point)
    Values = RSSI signal strength (-100 for missing APs)

    Args:
        db_path: path to SQLite database containing Wi-Fi scans
    
    Returns:
        fingerprints: DataFrame with BSSIDs as columns, scans as rows
        labels: Series of room labels for each scan
    """
    conn = sqlite3.connect(db_path)

    # Get all scan data joined with metadata
    query = """
        SELECT 
            sm.scan_id,
            sm.location,
            ws.bssid,
            ws.rssi
        FROM scan_metadata sm
        JOIN wifi_scan ws ON sm.scan_id = ws.scan_id
        WHERE sm.location IS NOT NULL AND sm.location != ''
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        raise ValueError("No data found in database. Check your table names and data.")

    print(f"[Preprocess] Loaded {len(df)} RSSI readings")
    print(f"[Preprocess] Unique scans: {df['scan_id'].nunique()}")
    print(f"[Preprocess] Unique APs (BSSIDs): {df['bssid'].nunique()}")
    print(f"[Preprocess] Rooms: {df['location'].nunique()} — {df['location'].unique().tolist()}")

    # Pivot: rows = scans, columns = BSSIDs, values = RSSI
    fingerprints = df.pivot_table(
        index='scan_id',
        columns='bssid',
        values='rssi',
        aggfunc='mean'  # average if duplicate BSSIDs in same scan
    )

    # Fill missing APs with -100 (not detected)
    fingerprints = fingerprints.fillna(-100)

    # Get labels (one per scan)
    labels = df.drop_duplicates('scan_id').set_index('scan_id')['location']
    labels = labels.reindex(fingerprints.index)

    print(f"[Preprocess] Fingerprint matrix shape: {fingerprints.shape}")
    print(f"[Preprocess] Samples per room:")
    for room, count in labels.value_counts().items():
        print(f"  {room}: {count}")

    return fingerprints, labels


def filter_low_variance_aps(fingerprints, min_detection_rate=0.05):
    """
    Remove APs that are detected in fewer than min_detection_rate of scans.
    These APs add noise without useful signal.
    
    Args:
        fingerprints: DataFrame of RSSI values (rows=scans, columns=BSSIDs)
        min_detection_rate: minimum fraction of scans in which AP must be detected to keep it (avoiding noisy features),
                            default value of 0.05 means AP must be detected in at least 5% of scans to be kept

    Returns:
        Filtered DataFrame with only APs that meet the detection threshold
    """
    detection_rate = (fingerprints > -100).mean()
    keep_aps = detection_rate[detection_rate >= min_detection_rate].index
    removed = len(fingerprints.columns) - len(keep_aps)

    print(f"[Preprocess] Removed {removed} low-variance APs (detected in <{min_detection_rate*100}% of scans)")
    print(f"[Preprocess] Remaining APs: {len(keep_aps)}")

    return fingerprints[keep_aps]


def normalise_rssi(fingerprints):
    """
    Min-max normalise RSSI values to [0, 1].
    -100 (not detected) -> 0.0
    0 (max signal) -> 1.0

    Args:
        fingerprints: DataFrame of RSSI values (rows=scans, columns=BSSIDs)

    Returns:
        normalised: DataFrame of normalised RSSI values
        scaler: fitted MinMaxScaler (for transforming new data with the same scaling)
    """
    scaler = MinMaxScaler()
    normalised_df = pd.DataFrame(
        scaler.fit_transform(fingerprints),
        index=fingerprints.index,
        columns=fingerprints.columns
    )
    return normalised_df, scaler


def load_and_preprocess(db_path=DEFAULT_DB_PATH, test_size=0.2, random_state=42, min_detection_rate=0.05):
    """
    Full preprocessing pipeline:
    1. Load fingerprints from SQLite
    2. Filter low-variance APs
    3. Normalise RSSI to [0, 1]
    4. Encode room labels
    5. Train/test split (stratified)

    Args:
        db_path: path to SQLite database
        test_size: fraction of data to reserve for testing (default: 0.2)
        random_state: random seed for reproducibility (default: 42)
        min_detection_rate: minimum fraction of scans in which AP must be detected to keep it (default: 0.05)
    
    Returns:
        X_train, X_test: numpy arrays of normalised fingerprints
        y_train, y_test: numpy arrays of encoded labels
        feature_names: list of BSSID column names
        label_encoder: fitted LabelEncoder (for decoding predictions)
        scaler: fitted MinMaxScaler (for transforming new data)
    """
    print("=" * 50)
    print("  Preprocessing Pipeline")
    print("=" * 50)

    fingerprints, labels = load_fingerprints_from_db(db_path)
    fingerprints = filter_low_variance_aps(fingerprints, min_detection_rate)
    fingerprints_norm, scaler = normalise_rssi(fingerprints)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print(f"\n[Preprocess] Label mapping:")
    for i, room in enumerate(label_encoder.classes_):
        print(f"  {i} -> {room}")

    X_train, X_test, y_train, y_test = train_test_split(
        fingerprints_norm.values,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\n[Preprocess] Train set: {X_train.shape[0]} samples")
    print(f"[Preprocess] Test set:  {X_test.shape[0]} samples")
    print(f"[Preprocess] Features:  {X_train.shape[1]} APs")
    print("=" * 50)

    feature_names = fingerprints_norm.columns.tolist()

    return X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler


def load_and_preprocess_temporal(db_path=DEFAULT_DB_PATH, test_ratio=0.2, min_detection_rate=0.05):
    """
    Temporal split: train on earlier scans, test on later scans.
    
    This tests whether the model generalises over TIME, not just
    across random samples. If accuracy drops significantly compared
    to random split, it indicates data leakage in the random split.
    
    For each room, the first 80% of scans (chronologically) go to
    training, and the last 20% go to testing.

    Args:
        db_path: path to SQLite database
        test_ratio: fraction of scans per room to reserve for testing (default: 0.2)
        min_detection_rate: minimum fraction of scans in which AP must be detected to keep it (default: 0.05)

    Returns:
        X_train, X_test: numpy arrays of normalised fingerprints
        y_train, y_test: numpy arrays of encoded labels
        feature_names: list of BSSID column names
        label_encoder: fitted LabelEncoder (for decoding predictions)
        scaler: fitted MinMaxScaler (for transforming new data)
    """
    print("=" * 50)
    print("  Preprocessing Pipeline (TEMPORAL SPLIT)")
    print("=" * 50)

    conn = sqlite3.connect(db_path)

    # Get scan data WITH timestamp for ordering
    query = """
        SELECT 
            sm.scan_id,
            sm.location,
            sm.timestamp,
            ws.bssid,
            ws.rssi
        FROM scan_metadata sm
        JOIN wifi_scan ws ON sm.scan_id = ws.scan_id
        WHERE sm.location IS NOT NULL AND sm.location != ''
        ORDER BY sm.timestamp
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"[Temporal] Loaded {len(df)} RSSI readings")

    fingerprints = df.pivot_table(
        index='scan_id', columns='bssid', values='rssi', aggfunc='mean'
    ).fillna(-100)

    scan_info = df.drop_duplicates('scan_id').set_index('scan_id')[['location', 'timestamp']]
    scan_info = scan_info.reindex(fingerprints.index)

    fingerprints = filter_low_variance_aps(fingerprints, min_detection_rate)
    fingerprints_norm, scaler = normalise_rssi(fingerprints)

    label_encoder = LabelEncoder()
    all_labels = label_encoder.fit_transform(scan_info['location'])

    # Temporal split: per room, first 80% -> train, last 20% -> test
    train_indices = []
    test_indices = []

    print(f"\n[Temporal] Splitting per room (train on early scans, test on late scans):")
    for room in sorted(scan_info['location'].unique()):
        room_mask = scan_info['location'] == room
        room_indices = np.where(room_mask)[0]

        # These are already sorted by timestamp from the SQL query
        split_point = int(len(room_indices) * (1 - test_ratio))
        train_indices.extend(room_indices[:split_point])
        test_indices.extend(room_indices[split_point:])

        print(f"  {room}: {split_point} train, {len(room_indices) - split_point} test")

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    X_train = fingerprints_norm.values[train_indices]
    X_test = fingerprints_norm.values[test_indices]
    y_train = all_labels[train_indices]
    y_test = all_labels[test_indices]

    feature_names = fingerprints_norm.columns.tolist()

    print(f"\n[Temporal] Train set: {X_train.shape[0]} samples (earlier scans)")
    print(f"[Temporal] Test set:  {X_test.shape[0]} samples (later scans)")
    print(f"[Temporal] Features:  {X_train.shape[1]} APs")

    # Show time ranges
    train_timestamps = scan_info.iloc[train_indices]['timestamp']
    test_timestamps = scan_info.iloc[test_indices]['timestamp']
    print(f"\n[Temporal] Train time range: {train_timestamps.min()} -> {train_timestamps.max()}")
    print(f"[Temporal] Test time range:  {test_timestamps.min()} -> {test_timestamps.max()}")
    print("=" * 50)

    return X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler
