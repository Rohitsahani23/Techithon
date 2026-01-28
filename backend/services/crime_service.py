# Data processing logic
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from config import Config
import os

def load_data():
    """Load and clean crime data from CSV"""
    try:
        # Handle relative path
        if not os.path.exists(Config.DATA_PATH):
            raise FileNotFoundError(f"Data file not found at {Config.DATA_PATH}")
        
        df = pd.read_csv(Config.DATA_PATH)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Rename common variations
        column_map = {
            "lat": "latitude",
            "lng": "longitude",
            "long": "longitude"
        }
        df.rename(columns=column_map, inplace=True)
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def get_all_crimes(filters=None):
    """Get all crime records with optional filtering"""
    df = load_data()
    
    if df.empty:
        return []
    
    # Apply filters if provided
    if filters:
        if 'crime_type' in filters and filters['crime_type']:
            df = df[df['crime_type'].isin(filters['crime_type'])]
        
        if 'risk_level' in filters and filters['risk_level']:
            df = df[df['severity'].isin(filters['risk_level'])]
        
        if 'date_from' in filters and filters['date_from']:
            df = df[df['date'] >= filters['date_from']]
        
        if 'date_to' in filters and filters['date_to']:
            df = df[df['date'] <= filters['date_to']]
    
    crimes = df.to_dict(orient="records")
    
    # Convert to JSON-serializable format
    for crime in crimes:
        for key, value in crime.items():
            if pd.isna(value):
                crime[key] = None
            elif isinstance(value, np.integer):
                crime[key] = int(value)
            elif isinstance(value, np.floating):
                crime[key] = float(value)
    
    return crimes


def get_hotspots(filters=None):
    """Get crime hotspots using KMeans clustering"""
    df = load_data()
    
    if df.empty:
        return []
    
    # Apply filters if provided
    if filters:
        if 'crime_type' in filters and filters['crime_type']:
            df = df[df['crime_type'].isin(filters['crime_type'])]
    
    # Check for required columns
    if not {'latitude', 'longitude'}.issubset(df.columns):
        return {"error": "CSV must contain latitude and longitude columns"}
    
    coords = df[['latitude', 'longitude']].values
    
    # Ensure we have enough data points
    n_clusters = min(4, len(df))
    
    if n_clusters < 1:
        return []
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['hotspot'] = kmeans.fit_predict(coords)
    
    hotspots = []
    for cluster in df['hotspot'].unique():
        cluster_data = df[df['hotspot'] == cluster]
        
        # Get crime type and last date
        crime_type = cluster_data['crime_type'].mode()[0] if len(cluster_data) > 0 else 'Mixed'
        last_date = cluster_data['date'].max() if len(cluster_data) > 0 else None
        
        hotspots.append({
            "hotspot_id": int(cluster),
            "crime_count": int(len(cluster_data)),
            "center_lat": float(cluster_data['latitude'].mean()),
            "center_lng": float(cluster_data['longitude'].mean()),
            "crime_type": str(crime_type),
            "last_date": str(last_date) if last_date else None
        })
    
    # Sort by crime count descending
    hotspots.sort(key=lambda x: x['crime_count'], reverse=True)
    
    return hotspots


def get_risk_analysis(filters=None):
    """Analyze risk levels by area"""
    df = load_data()
    
    if df.empty:
        return []
    
    # Apply filters if provided
    if filters:
        if 'crime_type' in filters and filters['crime_type']:
            df = df[df['crime_type'].isin(filters['crime_type'])]
    
    # Map severity to risk scores
    severity_map = {"low": 1, "medium": 2, "high": 3}
    df['risk_score_value'] = df['severity'].str.lower().map(severity_map)
    
    # Calculate risk by area
    area_risk = df.groupby("area").agg({
        'risk_score_value': 'mean',
        'id': 'count'
    }).reset_index()
    area_risk.columns = ['area', 'risk_score', 'crime_count']
    
    result = []
    for _, row in area_risk.iterrows():
        risk_score = float(row['risk_score'])
        
        # Determine risk level
        if risk_score > 2.2:
            level = "High"
        elif risk_score > 1.5:
            level = "Medium"
        else:
            level = "Low"
        
        # Normalize risk score to 0-100
        normalized_score = (risk_score / 3.0) * 100
        
        result.append({
            "area": str(row['area']),
            "risk_level": level,
            "risk_score": round(normalized_score, 2),
            "crime_count": int(row['crime_count'])
        })
    
    # Sort by risk score descending
    result.sort(key=lambda x: x['risk_score'], reverse=True)
    
    return result


def get_crime_stats():
    """Get overall crime statistics"""
    df = load_data()
    
    if df.empty:
        return {
            "total_crimes": 0,
            "total_areas": 0,
            "avg_crimes_per_area": 0,
            "crime_by_type": []
        }
    
    total_crimes = len(df)
    total_areas = df['area'].nunique()
    avg_crimes = total_crimes / total_areas if total_areas > 0 else 0
    
    # Crime by type
    crime_by_type = df['crime_type'].value_counts().to_dict()
    
    return {
        "total_crimes": int(total_crimes),
        "total_areas": int(total_areas),
        "avg_crimes_per_area": round(float(avg_crimes), 2),
        "crime_by_type": crime_by_type
    }

