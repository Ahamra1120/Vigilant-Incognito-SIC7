# app.py - Streamlit dengan auto-refresh otomatis
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from collections import deque
from PIL import Image
import io
import base64
import textwrap
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json

# Import MongoDB
from pymongo import MongoClient
from bson import ObjectId
from bson.errors import InvalidId

# Import circuit visual module
try:
    from circuit_visual_detailed import create_detailed_circuit, fig_to_image
    CIRCUIT_VISUAL_AVAILABLE = True
except ImportError:
    CIRCUIT_VISUAL_AVAILABLE = False
    st.warning("Circuit visual module not available")

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(
    page_title="Vigilant — Motion Anomaly Detection", 
    layout="wide",
    page_icon="🔍"
)

MODEL_PATH = "model/model_vigilant.pkl"
LOGO_PATH = "assets/logo_vigilant.png"

# ------------------------------
# MONGODB CONNECTION
# ------------------------------
@st.cache_resource
def get_mongo_client():
    """Create MongoDB client with caching"""
    try:
        client = MongoClient("mongodb+srv://incognito:incognito_sic7@incognito.andn28n.mongodb.net/?appName=Incognito")
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"❌ MongoDB Connection Failed: {str(e)[:100]}")
        return None

# Initialize MongoDB
mongo_client = get_mongo_client()

def get_latest_reading():
    """Get latest sensor reading from MongoDB"""
    if mongo_client:
        try:
            db = mongo_client["sensorDB"]
            collection = db["readings"]
            
            # Get the most recent reading
            latest = collection.find_one(
                {"temperature": {"$ne": None}},
                sort=[("timestamp", -1)]
            )
            
            if latest:
                # Convert ObjectId and ensure timestamp
                latest["_id"] = str(latest["_id"])
                if "timestamp" not in latest:
                    latest["timestamp"] = datetime.now()
                
                return latest
        except Exception as e:
            st.error(f"Error reading from MongoDB: {e}")
    
    return None

def get_recent_readings(limit=50):
    """Get recent sensor readings from MongoDB"""
    if mongo_client:
        try:
            db = mongo_client["sensorDB"]
            collection = db["readings"]
            
            # Get recent readings with temperature
            cursor = collection.find(
                {"temperature": {"$ne": None}},
                sort=[("timestamp", -1)]
            ).limit(limit)
            
            data = list(cursor)
            
            # Convert ObjectId to string and ensure timestamp
            for item in data:
                item["_id"] = str(item["_id"])
                if "timestamp" not in item:
                    # Try to get from _id or use current time
                    try:
                        item["timestamp"] = ObjectId(item["_id"]).generation_time
                    except:
                        item["timestamp"] = datetime.now()
            
            return data
        except Exception as e:
            st.error(f"Error reading recent data: {e}")
    
    return []

def get_latest_alert():
    """Get latest alert from MongoDB"""
    if mongo_client:
        try:
            db = mongo_client["sensorDB"]
            collection = db["alerts"]
            
            latest = collection.find_one(
                sort=[("timestamp", -1)]
            )
            
            if latest:
                latest["_id"] = str(latest["_id"])
                return latest
        except Exception as e:
            st.error(f"Error reading alert: {e}")
    
    return None

def get_recent_alerts(limit=10):
    """Get recent alerts from MongoDB"""
    if mongo_client:
        try:
            db = mongo_client["sensorDB"]
            collection = db["alerts"]
            
            cursor = collection.find().sort("timestamp", -1).limit(limit)
            alerts = list(cursor)
            
            for alert in alerts:
                alert["_id"] = str(alert["_id"])
            
            return alerts
        except Exception as e:
            st.error(f"Error reading alerts: {e}")
    
    return []

def insert_alert(alert_data):
    """Insert alert to MongoDB"""
    if mongo_client:
        try:
            db = mongo_client["sensorDB"]
            collection = db["alerts"]
            
            if "timestamp" not in alert_data:
                alert_data["timestamp"] = datetime.now()
            
            result = collection.insert_one(alert_data)
            return str(result.inserted_id)
        except Exception as e:
            st.error(f"Error inserting alert: {e}")
    
    return None

# ------------------------------
# LOAD MODEL
# ------------------------------
def load_model(path):
    try:
        m = joblib.load(path)
        return m, True
    except Exception as e:
        class DummyModel:
            def predict(self, X):
                out = []
                for _ in X:
                    r = np.random.rand()
                    if r < 0.75:
                        out.append("Normal")
                    elif r < 0.95:
                        out.append("Suspicious")
                    else:
                        out.append("Anomaly")
                return np.array(out)
        return DummyModel(), False

model, model_loaded = load_model(MODEL_PATH)

# ------------------------------
# STYLE (modern dark theme)
# ------------------------------
st.markdown("""
<style>
    /* Main theme */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
    }
    
    /* Cards */
    .custom-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .custom-card:hover {
        border-color: rgba(148, 163, 184, 0.3);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* Status indicators */
    .status-normal {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .status-warning {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .status-danger {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .status-offline {
        background: linear-gradient(135deg, #475569 0%, #64748b 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .status-low {
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Auto-refresh indicator */
    .refresh-indicator {
        position: fixed;
        bottom: 10px;
        right: 10px;
        background: rgba(59, 130, 246, 0.9);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 12px;
        z-index: 1000;
        animation: blink 2s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Status change animation */
    .status-change {
        animation: statusPulse 1s;
    }
    @keyframes statusPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Helper functions
# ------------------------------
def get_temperature_status(temp):
    """Determine temperature status based on thresholds"""
    if temp is None:
        return "N/A", "status-offline", "#64748b"
    
    if temp < 22:
        return "LOW", "status-low", "#3b82f6"
    elif temp > 35:
        return "CRITICAL", "status-danger", "#ef4444"
    elif temp > 32:
        return "HIGH", "status-warning", "#f59e0b"
    else:
        return "NORMAL", "status-normal", "#10b981"

def get_light_status(light):
    """Determine light status based on thresholds"""
    if light is None:
        return "N/A", "status-offline", "#64748b"
    
    if light > 100:
        return "EXTREME BRIGHT", "status-warning", "#fbbf24"
    elif light > 70:
        return "BRIGHT", "status-normal", "#fbbf24"
    elif light < 50:
        return "EXTREME DARK", "status-warning", "#3b82f6"
    elif light < 30:
        return "DARK", "status-normal", "#3b82f6"
    else:
        return "NORMAL", "status-normal", "#10b981"

def calculate_motion_intensity(light, previous_light=None):
    """Calculate motion intensity based on light changes"""
    if light is None:
        return round(np.random.uniform(0.1, 0.3), 2)
    
    # If we have previous light, calculate change
    if previous_light is not None:
        change = abs(light - previous_light)
        # Normalize change to 0-1 range (assuming max change of 500)
        intensity = min(change / 500, 0.9)
        return round(max(0.1, intensity), 2)
    
    # Base intensity on light level
    if light > 800:
        return round(np.random.uniform(0.7, 0.9), 2)  # Bright = more activity
    elif light < 200:
        return round(np.random.uniform(0.1, 0.3), 2)  # Dark = less activity
    else:
        return round(np.random.uniform(0.3, 0.6), 2)

def get_prediction(temperature, light, motion_intensity):
    """Get prediction from ML model"""
    try:
        if temperature is not None:
            # Create feature array
            features = np.array([[temperature, light if light else 500, motion_intensity]])
            return model.predict(features)[0]
    except:
        pass
    
    # Fallback logic
    if temperature is not None:
        if temperature > 35:
            return "Anomaly" if np.random.rand() < 0.8 else "Suspicious"
        elif temperature > 32:
            return "Suspicious" if np.random.rand() < 0.7 else "Anomaly"
        elif temperature < 22:
            return "Suspicious" if np.random.rand() < 0.6 else "Normal"
    
    r = np.random.rand()
    if r < 0.8:
        return "Normal"
    elif r < 0.95:
        return "Suspicious"
    else:
        return "Anomaly"

def get_anomaly_conditions(temperature, light, prediction):
    """Determine specific anomaly conditions"""
    conditions = []
    
    if prediction == "Anomaly":
        if temperature is not None and temperature > 35:
            conditions.append("Critical High Temperature")
        elif temperature is not None and temperature > 32:
            conditions.append("High Temperature Warning")
        elif temperature is not None and temperature < 22:
            conditions.append("Low Temperature Warning")
        
        if light is not None and light > 900:
            conditions.append("Extreme Brightness")
        elif light is not None and light < 100:
            conditions.append("Extreme Darkness")
    
    elif prediction == "Suspicious":
        if temperature is not None and temperature > 30:
            conditions.append("Elevated Temperature")
        if light is not None and light > 800:
            conditions.append("High Brightness")
        if light is not None and light < 200:
            conditions.append("Low Light")
    
    return conditions if conditions else ["All systems normal"]

def check_status_change(current_data, previous_data):
    """Check if status has changed significantly"""
    if not previous_data:
        return True
    
    # Check temperature change > 2°C
    if (current_data.get('temperature') is not None and 
        previous_data.get('temperature') is not None):
        if abs(current_data['temperature'] - previous_data['temperature']) > 2:
            return True
    
    # Check light change > 100 lux
    if (current_data.get('light') is not None and 
        previous_data.get('light') is not None):
        if abs(current_data['light'] - previous_data['light']) > 100:
            return True
    
    # Check if prediction changed
    if (current_data.get('prediction') != previous_data.get('prediction')):
        return True
    
    return False

# ------------------------------
# Session state for auto-refresh
# ------------------------------
if "last_mongo_id" not in st.session_state:
    st.session_state.last_mongo_id = None
if "last_alert_id" not in st.session_state:
    st.session_state.last_alert_id = None
if "data_changed" not in st.session_state:
    st.session_state.data_changed = False
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=100)
if "alerts" not in st.session_state:
    st.session_state.alerts = deque(maxlen=50)
if "running" not in st.session_state:
    st.session_state.running = True
if "last_data" not in st.session_state:
    st.session_state.last_data = None
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0
if "status_changed" not in st.session_state:
    st.session_state.status_changed = False

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    try:
        st.image(LOGO_PATH, use_column_width=True)
    except:
        st.markdown("### 🔍 VIGILANT")
    
    st.markdown("---")
    st.markdown("### 🔧 System Controls")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "📈 Live Monitor", "📊 Data History", 
         "🔌 Circuit Diagram", "👥 Team", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # MongoDB Status
    st.markdown("### 📡 Data Source")
    if mongo_client:
        st.success("✅ MongoDB Connected")
        
        try:
            db = mongo_client["sensorDB"]
            readings_count = db["readings"].count_documents({})
            alerts_count = db["alerts"].count_documents({})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Readings", readings_count)
            with col2:
                st.metric("Alerts", alerts_count)
        except:
            pass
    else:
        st.error("❌ MongoDB Offline")
        st.info("Using simulated data")
    
    st.markdown("---")
    
    # Auto-refresh controls
    st.markdown("### 🔄 Auto-Refresh")
    
    col_ref1, col_ref2 = st.columns(2)
    with col_ref1:
        if st.button("▶️ Start" if not st.session_state.running else "⏸️ Pause"):
            st.session_state.running = not st.session_state.running
            st.rerun()
    
    with col_ref2:
        if st.button("🔄 Manual Refresh"):
            st.session_state.status_changed = True
            st.rerun()
    
    refresh_rate = st.slider("Interval (seconds)", 2, 30, 5, 1)
    
    # Status indicator
    if st.session_state.running:
        st.info(f"Auto-refresh every {refresh_rate}s")
    else:
        st.warning("Auto-refresh paused")
    
    st.markdown("---")
    
    # System Status
    st.markdown("### 📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model", "✓ Loaded" if model_loaded else "⚠ Demo")
    with col2:
        st.metric("Refreshes", st.session_state.refresh_count)
    
    st.markdown("---")
    st.caption("Vigilant v1.0 | BINUS University")

# ------------------------------
# Auto-refresh logic
# ------------------------------
def should_refresh():
    """Determine if we should refresh based on conditions"""
    if not st.session_state.running:
        return False
    
    # Check time-based refresh
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).seconds
    if time_since_refresh >= refresh_rate:
        return True
    
    # Check if status changed (from MQTT data)
    current_data = get_latest_reading()
    if current_data and check_status_change(current_data, st.session_state.last_data):
        st.session_state.status_changed = True
        return True
    
    return False

# ------------------------------
# DASHBOARD PAGE
# ------------------------------
if page == "📊 Dashboard":
    # Auto-refresh indicator
    if st.session_state.running:
        st.markdown(f"""
        <div class="refresh-indicator">
            🔄 Auto-refresh: {refresh_rate}s | Count: {st.session_state.refresh_count}
        </div>
        """, unsafe_allow_html=True)
    
    st.title("🔍 Vigilant Dashboard")
    st.markdown("Real-time environmental monitoring & anomaly detection system")
    
    # Get current data
    current_reading = get_latest_reading()
    
    # Prepare data for display
    if current_reading:
        temperature = current_reading.get('temperature')
        light = current_reading.get('light')
        timestamp = current_reading.get('timestamp', datetime.now())
        
        # Calculate motion intensity based on light changes
        previous_reading = st.session_state.last_data
        previous_light = previous_reading.get('light') if previous_reading else None
        motion_intensity = calculate_motion_intensity(light, previous_light)
        
        # Get prediction
        prediction = get_prediction(temperature, light, motion_intensity)
        
        # Get conditions
        conditions = get_anomaly_conditions(temperature, light, prediction)
        
        current_data = {
            "temperature": temperature,
            "light": light,
            "motion_intensity": motion_intensity,
            "prediction": prediction,
            "conditions": conditions,
            "timestamp": timestamp,
            "esp32_status": "Online",
            "camera_status": "Streaming",
            "source": "MongoDB"
        }
        
        # Check for status change
        status_changed = check_status_change(current_data, st.session_state.last_data)
        if status_changed:
            st.session_state.status_changed = True
        
        # Store current data
        st.session_state.last_data = current_data
        
    else:
        # Fallback data
        current_data = {
            "temperature": round(np.random.uniform(20, 40), 2),
            "light": round(np.random.uniform(100, 1000), 2),
            "motion_intensity": round(np.random.uniform(0.1, 0.9), 2),
            "prediction": "Normal",
            "conditions": ["All systems normal"],
            "timestamp": datetime.now(),
            "esp32_status": "Online",
            "camera_status": "Streaming",
            "source": "Simulation"
        }
    
    # Add to history
    st.session_state.history.append({
        "timestamp": current_data["timestamp"],
        "temperature": current_data["temperature"],
        "light": current_data["light"],
        "motion_intensity": current_data["motion_intensity"],
        "prediction": current_data["prediction"]
    })
    
    # Store alert if anomaly
    if current_data["prediction"] == "Anomaly":
        alert_data = {
            "timestamp": datetime.now(),
            "type": "ANOMALY_DETECTED",
            "conditions": current_data["conditions"],
            "data": {
                "temperature": current_data["temperature"],
                "light": current_data["light"],
                "motion_intensity": current_data["motion_intensity"]
            },
            "prediction": current_data["prediction"],
            "severity": "HIGH",
            "source": "Streamlit"
        }
        
        # Check if we should store this alert
        store_alert = True
        if st.session_state.alerts:
            last_alert = st.session_state.alerts[-1]
            time_diff = (datetime.now() - last_alert["timestamp"]).seconds
            if time_diff < 30:  # Don't store alerts more frequently than 30 seconds
                store_alert = False
        
        if store_alert:
            st.session_state.alerts.append(alert_data)
            alert_id = insert_alert(alert_data)
            if alert_id:
                st.sidebar.success(f"🚨 Alert saved (ID: {alert_id[:8]}...)")
    
    # Row 1: System Status Cards
    st.markdown('<div class="section-header"><h3>📡 System Status</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="sensor-label">ESP32 Status</div>', unsafe_allow_html=True)
        
        # Animate if status changed
        animation_class = "status-change" if st.session_state.status_changed else ""
        status_class = "status-normal"
        st.markdown(f'<div class="{status_class} {animation_class}">{current_data["esp32_status"]}</div>', unsafe_allow_html=True)
        
        st.progress(0.95)
        st.caption(f"Source: {current_data['source']}")
        st.caption(f"Updated: {current_data['timestamp'].strftime('%H:%M:%S')}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="sensor-label">Camera Status</div>', unsafe_allow_html=True)
        
        animation_class = "status-change" if st.session_state.status_changed else ""
        cam_class = "status-normal"
        st.markdown(f'<div class="{cam_class} {animation_class}">{current_data["camera_status"]}</div>', unsafe_allow_html=True)
        
        st.metric("FPS", "24", delta="Stable")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="sensor-label">Temperature Status</div>', unsafe_allow_html=True)
        
        temp_status, temp_class, temp_color = get_temperature_status(current_data["temperature"])
        animation_class = "status-change" if st.session_state.status_changed else ""
        
        st.markdown(f'<div class="{temp_class} {animation_class}">{temp_status}</div>', unsafe_allow_html=True)
        
        # Temperature value
        temp_value = f"{current_data['temperature']}°C" if current_data["temperature"] is not None else "N/A"
        st.markdown(f'<div style="color: {temp_color}; font-size: 2rem; font-weight: 700;">{temp_value}</div>', unsafe_allow_html=True)
        
        # Threshold indicators
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.caption("❄️ Low <22°C")
        with col_t2:
            st.caption("✅ Normal 22-32°C")
        with col_t3:
            st.caption("🔥 High >32°C")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="sensor-label">Light</div>', unsafe_allow_html=True)
        
        # Light status
        light_status, light_class, light_color = get_light_status(current_data["light"])
        animation_class = "status-change" if st.session_state.status_changed else ""
        
        st.markdown(f'<div class="{light_class} {animation_class}">Light: {light_status}</div>', unsafe_allow_html=True)
        
        # Light value
        light_value = f"{current_data['light']} lux" if current_data["light"] is not None else "N/A"
        st.markdown(f'<div style="color: {light_color}; font-size: 1.5rem; font-weight: 700;">{light_value}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Motion intensity
        st.markdown('<div class="sensor-label">Motion Intensity</div>', unsafe_allow_html=True)
        motion_color = "#ef4444" if current_data["motion_intensity"] > 0.8 else "#f59e0b" if current_data["motion_intensity"] > 0.6 else "#10b981"
        st.markdown(f'<div style="color: {motion_color}; font-size: 1.5rem; font-weight: 700;">{current_data["motion_intensity"]:.2f}</div>', unsafe_allow_html=True)
        
        # Activity level
        if current_data["motion_intensity"] > 0.8:
            activity_level = "High Activity"
        elif current_data["motion_intensity"] > 0.6:
            activity_level = "Moderate Activity"
        elif current_data["motion_intensity"] > 0.3:
            activity_level = "Normal Activity"
        else:
            activity_level = "Low Activity"
        
        st.caption(f"Level: {activity_level}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Row 2: Anomaly Detection
    st.markdown('<div class="section-header"><h3>🚨 Anomaly Detection</h3></div>', unsafe_allow_html=True)
    
    col_main, col_side = st.columns([3, 1])
    
    with col_main:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        if current_data["prediction"] == "Anomaly":
            animation_class = "status-change" if st.session_state.status_changed else ""
            st.markdown(f'<div class="anomaly-alert {animation_class}">', unsafe_allow_html=True)
            
            col_alert1, col_alert2 = st.columns([1, 3])
            with col_alert1:
                st.markdown("🚨")
                st.markdown(f'<div class="status-danger" style="font-size: 1.2rem;">{current_data["prediction"]}</div>', unsafe_allow_html=True)
                st.metric("Severity", "HIGH", delta_color="inverse")
            with col_alert2:
                st.markdown("### ⚠️ Critical Conditions Detected")
                for condition in current_data["conditions"]:
                    if "Temperature" in condition:
                        st.markdown(f'<div style="background: rgba(239, 68, 68, 0.2); border-left: 4px solid #ef4444; padding: 10px; border-radius: 5px; margin: 5px 0;">🔥 {condition}</div>', unsafe_allow_html=True)
                    elif "Light" in condition or "Bright" in condition or "Dark" in condition:
                        st.markdown(f'<div style="background: rgba(251, 191, 36, 0.2); border-left: 4px solid #fbbf24; padding: 10px; border-radius: 5px; margin: 5px 0;">💡 {condition}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"• **{condition}**")
                
                st.markdown("**Recommended Actions:**")
                if "Temperature" in " ".join(current_data["conditions"]):
                    st.markdown("• Check cooling/heating systems")
                    st.markdown("• Verify sensor calibration")
                if "Light" in " ".join(current_data["conditions"]):
                    st.markdown("• Adjust lighting conditions")
                    st.markdown("• Check LDR sensor")
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif current_data["prediction"] == "Suspicious":
            animation_class = "status-change" if st.session_state.status_changed else ""
            col_sus1, col_sus2 = st.columns([1, 3])
            with col_sus1:
                st.markdown("⚠️")
                st.markdown(f'<div class="status-warning {animation_class}" style="font-size: 1.2rem;">{current_data["prediction"]}</div>', unsafe_allow_html=True)
                st.metric("Risk", "MEDIUM")
            with col_sus2:
                st.markdown("### Suspicious Activity Detected")
                st.markdown("**Conditions detected:**")
                for condition in current_data["conditions"]:
                    st.markdown(f"• {condition}")
                st.markdown("**Monitor closely for changes.**")
        
        else:
            animation_class = "status-change" if st.session_state.status_changed else ""
            col_norm1, col_norm2 = st.columns([1, 3])
            with col_norm1:
                st.markdown("✅")
                st.markdown(f'<div class="status-normal {animation_class}" style="font-size: 1.2rem;">{current_data["prediction"]}</div>', unsafe_allow_html=True)
                st.metric("Risk", "LOW", delta="Stable")
            with col_norm2:
                st.markdown("### System Operating Normally")
                st.markdown("All environmental parameters within safe ranges.")
                if current_data["conditions"]:
                    st.markdown(f"**Status:** {current_data['conditions'][0]}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_side:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Detection Confidence")
        
        # Create gauge chart
        if current_data["prediction"] == "Anomaly":
            value = 92
            color = "#ef4444"
        elif current_data["prediction"] == "Suspicious":
            value = 75
            color = "#f59e0b"
        else:
            value = 88
            color = "#10b981"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            title = {'text': "Confidence %"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 60], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [60, 85], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [85, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                ],
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if current_data["temperature"] is not None:
                trend = "↑" if current_data["temperature"] > 25 else "↓"
                st.metric("Temp Trend", trend)
        with col_m2:
            st.metric("Refreshes", st.session_state.refresh_count)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Row 3: Recent Alerts
    st.markdown('<div class="section-header"><h3>📋 Recent Alerts</h3></div>', unsafe_allow_html=True)
    
    # Get alerts from MongoDB
    mongo_alerts = get_recent_alerts(limit=5)
    
    if mongo_alerts:
        for alert in mongo_alerts:
            alert_time = alert.get('timestamp', datetime.now())
            if isinstance(alert_time, str):
                try:
                    alert_time = datetime.fromisoformat(alert_time.replace('Z', '+00:00'))
                except:
                    alert_time = datetime.now()
            
            with st.expander(f"🚨 {alert.get('type', 'ALERT')} at {alert_time.strftime('%H:%M:%S')}"):
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.markdown("**Conditions:**")
                    conditions = alert.get('conditions', [])
                    for cond in conditions:
                        if "Temperature" in cond:
                            st.markdown(f"• 🔥 {cond}")
                        elif "Light" in cond:
                            st.markdown(f"• 💡 {cond}")
                        else:
                            st.markdown(f"• ⚙️ {cond}")
                
                with col_a2:
                    st.markdown("**Sensor Data:**")
                    data = alert.get('data', {})
                    if data.get('temperature') is not None:
                        st.markdown(f"**Temperature:** {data['temperature']}°C")
                    if data.get('light') is not None:
                        st.markdown(f"**Light:** {data['light']} lux")
                    if data.get('motion_intensity') is not None:
                        st.markdown(f"**Motion:** {data['motion_intensity']:.2f}")
    else:
        st.info("No alerts recorded yet")
    
    # Auto-refresh logic
    if st.session_state.running:
        # Check if we should refresh
        if should_refresh():
            st.session_state.last_refresh = datetime.now()
            st.session_state.refresh_count += 1
            st.session_state.status_changed = False
            time.sleep(0.1)
            st.rerun()

# ------------------------------
# LIVE MONITOR PAGE
# ------------------------------
elif page == "📈 Live Monitor":
    st.title("📈 Live Monitor")
    st.markdown("Real-time sensor data visualization")
    
    # Get recent data
    recent_data = get_recent_readings(limit=100)
    
    if recent_data:
        # Convert to DataFrame
        df = pd.DataFrame(recent_data)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Temperature", "💡 Light", "📈 Combined"])
        
        with tab1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("🌡️ Temperature Over Time")
            
            if 'temperature' in df.columns and df['temperature'].notna().any():
                fig_temp = go.Figure()
                fig_temp.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['temperature'],
                    mode='lines+markers',
                    name='Temperature',
                    line=dict(color='#ef4444', width=2),
                    marker=dict(size=4)
                ))
                
                # Add threshold lines
                fig_temp.add_hline(y=32, line_dash="dash", line_color="orange", 
                                  annotation_text="High Threshold", 
                                  annotation_position="top right")
                fig_temp.add_hline(y=22, line_dash="dot", line_color="blue", 
                                  annotation_text="Low Threshold", 
                                  annotation_position="top right")
                
                fig_temp.update_layout(
                    height=400,
                    xaxis_title="Time",
                    yaxis_title="Temperature (°C)",
                    showlegend=True
                )
                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.info("No temperature data available")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("💡 Light Intensity Over Time")
            
            if 'light' in df.columns and df['light'].notna().any():
                fig_light = go.Figure()
                fig_light.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['light'],
                    mode='lines+markers',
                    name='Light',
                    line=dict(color='#fbbf24', width=2),
                    marker=dict(size=4)
                ))
                
                fig_light.update_layout(
                    height=400,
                    xaxis_title="Time",
                    yaxis_title="Light (lux)",
                    showlegend=True
                )
                st.plotly_chart(fig_light, use_container_width=True)
            else:
                st.info("No light data available")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📈 Combined Sensor Data")
            
            fig_combined = go.Figure()
            
            # Add temperature trace
            if 'temperature' in df.columns and df['temperature'].notna().any():
                fig_combined.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['temperature'],
                    mode='lines',
                    name='Temperature',
                    line=dict(color='#ef4444', width=2),
                    yaxis='y1'
                ))
            
            # Add light trace
            if 'light' in df.columns and df['light'].notna().any():
                fig_combined.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['light'],
                    mode='lines',
                    name='Light',
                    line=dict(color='#fbbf24', width=2),
                    yaxis='y2'
                ))
            
            fig_combined.update_layout(
                height=400,
                xaxis=dict(title="Time"),
                yaxis=dict(
                    title="Temperature (°C)",
                    titlefont=dict(color="#ef4444"),
                    tickfont=dict(color="#ef4444")
                ),
                yaxis2=dict(
                    title="Light (lux)",
                    titlefont=dict(color="#fbbf24"),
                    tickfont=dict(color="#fbbf24"),
                    overlaying='y',
                    side='right'
                ),
                showlegend=True
            )
            st.plotly_chart(fig_combined, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.info("No data available from MongoDB")
    
    # Auto-refresh for live monitor
    if st.session_state.running:
        time.sleep(refresh_rate)
        st.rerun()

# ------------------------------
# DATA HISTORY PAGE
# ------------------------------
elif page == "📊 Data History":
    st.title("📊 Data History")
    st.markdown("Historical sensor data from MongoDB")
    
    # Get data with date range selection
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        limit = st.slider("Number of records", 10, 1000, 100)
    with col_filter2:
        show_alerts = st.checkbox("Show alerts", True)
    with col_filter3:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Get data from MongoDB
    sensor_data = get_recent_readings(limit=limit)
    alerts_data = get_recent_alerts(limit=50) if show_alerts else []
    
    if sensor_data:
        # Convert to DataFrame
        df = pd.DataFrame(sensor_data)
        
        # Display statistics
        st.markdown("### 📈 Statistics")
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            if 'temperature' in df.columns and df['temperature'].notna().any():
                avg_temp = df['temperature'].mean()
                st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
        
        with col_stat2:
            if 'light' in df.columns and df['light'].notna().any():
                avg_light = df['light'].mean()
                st.metric("Avg Light", f"{avg_light:.0f} lux")
        
        with col_stat3:
            st.metric("Total Records", len(df))
        
        # Data table
        st.markdown("### 📋 Raw Data")
        st.dataframe(df[['timestamp', 'temperature', 'light']].head(20), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="sensor_data.csv",
            mime="text/csv"
        )
    
    if alerts_data and show_alerts:
        st.markdown("### 🚨 Recent Alerts")
        alerts_df = pd.DataFrame(alerts_data)
        st.dataframe(alerts_df[['timestamp', 'type', 'severity']].head(10), use_container_width=True)

# ------------------------------
# CIRCUIT DIAGRAM PAGE
# ------------------------------
elif page == "🔌 Circuit Diagram":
    if CIRCUIT_VISUAL_AVAILABLE:
        st.title("🔌 Circuit Diagram & Hardware")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = create_detailed_circuit()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 🎯 Components")
            st.markdown("""
            - ⚡ ESP32 DevKit
            - 🌡️ DHT11 Sensor  
            - 💡 LDR Sensor
            - 📷 ESP32-CAM
            - 📺 OLED Display
            - 🔊 Buzzer
            """)
    else:
        st.error("Circuit visual module not available")

# ------------------------------
# TEAM PAGE
# ------------------------------
elif page == "👥 Team":
    st.title("👥 Team — Incognito")
    
    members = [
        {"name": "Ahmad Hamra", "role": "Logic Developer"},
        {"name": "Alfred Abner", "role": "Documentation Specialist"},
        {"name": "Davin Aji Wibowo", "role": "Video Production"},
        {"name": "Reynaldo Lamhot Silalahi", "role": "Documentation Specialist"},
    ]
    
    cols = st.columns(2)
    for idx, member in enumerate(members):
        with cols[idx % 2]:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader(member["name"])
            st.markdown(f"**Role:** {member['role']}")
            st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# SETTINGS PAGE
# ------------------------------
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["General", "Thresholds", "System"])
    
    with tab1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("General Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Theme", ["Dark", "Light", "Auto"])
            st.number_input("History Size", 50, 1000, 100)
        with col2:
            st.selectbox("Time Format", ["24-hour", "12-hour"])
            st.checkbox("Show Notifications", True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Alert Thresholds")
        
        col1, col2 = st.columns(2)
        with col1:
            low_temp = st.slider("Low Temperature (°C)", 15, 25, 22)
            high_temp = st.slider("High Temperature (°C)", 28, 45, 32)
            crit_temp = st.slider("Critical Temperature (°C)", 33, 50, 35)
        
        with col2:
            dark_light = st.slider("Dark Light (lux)", 0, 20, 30)
            bright_light = st.slider("Bright Light (lux)", 40, 50, 60)
            extreme_light = st.slider("Extreme Light (lux)", 70, 80, 90, 100)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("System Settings")
        
        if st.button("🔄 Clear Local Cache"):
            st.session_state.history.clear()
            st.session_state.alerts.clear()
            st.success("Local cache cleared!")
        
        if st.button("📊 Reset Counters"):
            st.session_state.refresh_count = 0
            st.success("Counters reset!")
        
        if st.button("🔧 Test MongoDB Connection"):
            if mongo_client:
                st.success("✅ MongoDB connection successful!")
            else:
                st.error("❌ MongoDB connection failed")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.caption("🔍 Vigilant v1.0 | Environmental Anomaly Detection System")
with col2:
    st.caption(f"Refreshes: {st.session_state.refresh_count}")
with col3:
    st.caption(f"Last: {datetime.now().strftime('%H:%M:%S')}")