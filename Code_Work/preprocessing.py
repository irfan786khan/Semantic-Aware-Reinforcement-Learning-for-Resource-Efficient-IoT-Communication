# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_sigfox_data(sigfox_path, bs_mapping_path, output_path):
    # Load Sigfox Antwerp dataset
    sigfox_df = pd.read_csv(sigfox_path)

    # Clean column names (remove quotes and strip spaces)
    sigfox_df.columns = [col.strip().replace("'", "") for col in sigfox_df.columns]

    # Convert 'RX Time' to datetime, removing quotes first
    sigfox_df["RX Time"] = pd.to_datetime(sigfox_df["RX Time"].str.replace("'", ""), errors='coerce')

    # Extract hour from 'RX Time'
    sigfox_df["hour"] = sigfox_df["RX Time"].dt.hour

    # Identify RSSI columns: columns starting with 'BS '
    rssi_cols = [col for col in sigfox_df.columns if col.startswith("BS ")]

    # Replace -200 with NaN to exclude inactive stations
    rssi_data = sigfox_df[rssi_cols].replace(-200, np.nan)

    # Calculate mean RSSI (ignoring NaN)
    sigfox_df["mean_rssi"] = rssi_data.mean(axis=1)

    # Count number of active base stations (RSSI > -200)
    sigfox_df["num_active_bs"] = rssi_data.notna().sum(axis=1)

    # Load base station mapping (optional, could be used later)
    bs_mapping = pd.read_csv(bs_mapping_path)
    bs_mapping.columns = [col.strip() for col in bs_mapping.columns]

    # Select semantic features
    semantic_df = sigfox_df[["mean_rssi", "num_active_bs", "Latitude", "Longitude", "hour"]]

    # Save semantic features to CSV
    semantic_df.to_csv(output_path, index=False)

    print(f"✅ Semantic features saved to '{output_path}'")
    print(semantic_df.head())
    
    return semantic_df

def prepare_classification_data(data):
    """Prepare data for classification task"""
    features = ["mean_rssi", "num_active_bs", "Latitude", "Longitude", "hour"]
    
    # Add classification label
    data['label'] = ((data['mean_rssi'] > -110) & (data['num_active_bs'] >= 3)).astype(int)
    print(f"Label distribution:\n{data['label'].value_counts()}")
    
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data[features])
    
    return X_scaled, data['label'], features, scaler

def prepare_anomaly_data(data):
    """Prepare data for anomaly detection"""
    features = ["mean_rssi", "num_active_bs", "Latitude", "Longitude", "hour"]
    
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data[features])
    
    return X_scaled, features, scaler