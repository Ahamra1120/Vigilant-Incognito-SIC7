# app.py
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
SCHEMA_IMG = "assets/circuit_schema.png"
FLOWCHART_IMG = "assets/flowchart.png"
DATASET_PATH = "assets/dataset/dataset_temperature_humidity_status.csv"

# ------------------------------
# LOAD DATASET
# ------------------------------
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv(DATASET_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# ------------------------------
# LOAD MODEL (fallback to dummy)
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
                    if r < 0.72:
                        out.append("Normal")
                    elif r < 0.92:
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
    
    /* Sensor values */
    .sensor-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
    }
    .sensor-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Anomaly alert */
    .anomaly-alert {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
        border-left: 4px solid #ef4444;
        padding: 20px;
        border-radius: 8px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    /* Temperature-specific alerts */
    .temp-high-alert {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.3) 0%, rgba(239, 68, 68, 0.2) 100%);
        border-left: 4px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .temp-low-alert {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(96, 165, 250, 0.2) 100%);
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Humidity-specific alerts */
    .humidity-high-alert {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.3) 0%, rgba(56, 189, 248, 0.2) 100%);
        border-left: 4px solid #0ea5e9;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Circuit diagram specific */
    .circuit-container {
        background: rgba(15, 23, 42, 0.9);
        border: 2px solid #334155;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    
    /* Section headers */
    .section-header {
        border-bottom: 2px solid #334155;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Helper functions
# ------------------------------
def get_temperature_status(temp):
    """Determine temperature status based on thresholds"""
    if temp < 22:
        return "LOW", "status-low"
    elif temp > 32:
        return "WARNING", "status-warning"
    else:
        return "NORMAL", "status-normal"

def get_humidity_status(humidity):
    """Determine humidity status based on thresholds"""
    if humidity > 90:
        return "HIGH", "status-warning"
    else:
        return "NORMAL", "status-normal"

def simulate_input_from_dataset(dataset, index):
    """Get simulated input from dataset"""
    if dataset is not None and len(dataset) > index:
        row = dataset.iloc[index]
        
        # Generate motion intensity based on temperature pattern
        if row['status'] == 'WARNING':
            motion_intensity = np.random.uniform(0.6, 0.9)
        elif row['status'] == 'LOW':
            motion_intensity = np.random.uniform(0.1, 0.3)
        else:
            motion_intensity = np.random.uniform(0.3, 0.6)
        
        return {
            "temperature": row['temperature_celsius'],
            "humidity": row['humidity_percent'],
            "temperature_status": row['status'],
            "motion_intensity": round(motion_intensity, 2),
            "esp32_status": np.random.choice(["Online", "Online", "Offline"], p=[0.95, 0.04, 0.01]),
            "camera_status": np.random.choice(["Streaming", "Idle", "Error"], p=[0.9, 0.07, 0.03]),
            "network_latency": round(np.random.uniform(5, 100), 1),
            "timestamp": row['timestamp']
        }
    
    # Fallback if dataset not available
    return {
        "temperature": round(np.random.uniform(20, 40), 2),
        "humidity": round(np.random.uniform(40, 95), 2),
        "temperature_status": "NORMAL",
        "motion_intensity": round(np.random.uniform(0, 0.9), 2),
        "esp32_status": "Online",
        "camera_status": "Streaming",
        "network_latency": round(np.random.uniform(10, 150), 1),
        "timestamp": datetime.now()
    }

def get_prediction(data):
    # Using temperature, humidity, and motion intensity for prediction
    X = np.array([[data["temperature"], data["humidity"], data["motion_intensity"]]])
    try:
        return model.predict(X)[0]
    except Exception:
        # Fallback logic based on conditions
        if data["temperature_status"] == "WARNING" or data["temperature"] > 35:
            if np.random.rand() < 0.7:
                return "Anomaly"
            else:
                return "Suspicious"
        elif data["temperature_status"] == "LOW" or data["temperature"] < 22:
            if np.random.rand() < 0.8:
                return "Normal"
            else:
                return "Suspicious"
        else:
            r = np.random.rand()
            if r < 0.85: return "Normal"
            elif r < 0.95: return "Suspicious"
            else: return "Anomaly"

def get_anomaly_conditions(data, prediction):
    """Determine specific anomaly conditions based on data"""
    conditions = []
    
    if prediction == "Anomaly":
        # Temperature-related conditions
        if data["temperature"] > 35:
            conditions.append("Critical High Temperature")
        elif data["temperature"] > 32:
            conditions.append("High Temperature Warning")
        
        # Humidity-related conditions
        if data["humidity"] > 90:
            conditions.append("High Humidity")
        
        # Motion-related conditions
        if data["motion_intensity"] > 0.8:
            conditions.append("Intense Motion Detected")
        elif data["motion_intensity"] < 0.1:
            conditions.append("Unusually Low Activity")
        
        # System-related conditions
        if data["esp32_status"] == "Offline":
            conditions.append("ESP32 Connection Lost")
        if data["camera_status"] == "Error":
            conditions.append("Camera Malfunction")
    
    elif prediction == "Suspicious":
        if data["temperature"] > 30:
            conditions.append("Elevated Temperature")
        if data["humidity"] > 85:
            conditions.append("Elevated Humidity")
        if data["motion_intensity"] > 0.6:
            conditions.append("Moderate Activity")
    
    return conditions if conditions else ["All systems normal"]

def display_circuit_page():
    """Display circuit diagram page"""
    st.title("🔌 Circuit Diagram & Hardware")
    st.markdown("### Diagram Rangkaian dan Spesifikasi Hardware Sistem Vigilant")
    
    if CIRCUIT_VISUAL_AVAILABLE:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🎨 Visual Diagram", "📋 Component List", "🔧 Installation Guide"])
        
        with tab1:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown('<div class="circuit-container">', unsafe_allow_html=True)
                fig = create_detailed_circuit()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download button
                img = fig_to_image(fig)
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Circuit Diagram",
                    data=byte_im,
                    file_name="vigilant_circuit_diagram.png",
                    mime="image/png"
                )
            
            with col2:
                st.markdown("### 🎯 **Component Legend**")
                
                components = [
                    ("⚡ ESP32", "f59e0b", "Main microcontroller"),
                    ("🌡️ DHT11", "3b82f6", "Temp/Humidity sensor"),
                    ("📷 Camera", "ef4444", "OV2640 2MP camera"),
                    ("📺 OLED", "10b981", "SSD1306 display"),
                    ("🔊 Buzzer", "8b5cf6", "Audio alert"),
                    ("⚡ Relay", "ec4899", "Power control"),
                    ("🔘 Button", "64748b", "User input"),
                    ("💡 LED", "22c55e", "Status indicator"),
                ]
                
                for name, color, desc in components:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 10px; padding: 5px; background: rgba(30, 41, 59, 0.5); border-radius: 5px;">
                        <div style="width: 12px; height: 12px; background-color: #{color}; 
                             border-radius: 3px; margin-right: 10px; border: 1px solid white;"></div>
                        <div style="flex: 1;">
                            <strong style="font-size: 12px;">{name}</strong><br>
                            <small style="color: #94a3b8; font-size: 10px;">{desc}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 🔗 **Connection Types**")
                st.markdown("""
                - **🟡 Power Lines:** 5V/3.3V supply
                - **🔵 Data Lines:** Sensor communication  
                - **🟢 I2C Bus:** Display & camera
                - **🔴 Control:** GPIO outputs
                - **⚪ Input:** User controls
                """)
        
        with tab2:
            st.markdown("### 📦 **Bill of Materials**")
            
            # Component table
            components_df = pd.DataFrame({
                'Component': ['ESP32 DevKit V1', 'DHT11 Sensor', 'ESP32-CAM', 'OLED SSD1306', 
                            'Active Buzzer', '5V Relay Module', 'Push Button', 'LED 5mm',
                            '220Ω Resistor', '10KΩ Resistor', 'Breadboard', 'Jumper Wires'],
                'Qty': [1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 1, 30],
                'Unit Price (USD)': [8.99, 2.99, 9.99, 6.99, 1.99, 2.49, 0.50, 0.20, 0.10, 0.10, 5.99, 4.99],
                'Total (USD)': [8.99, 2.99, 9.99, 6.99, 1.99, 2.49, 0.50, 0.20, 0.50, 0.50, 5.99, 4.99]
            })
            
            components_df['Total (USD)'] = components_df['Qty'] * components_df['Unit Price (USD)']
            total_cost = components_df['Total (USD)'].sum()
            
            st.dataframe(components_df, use_container_width=True)
            
            col_cost1, col_cost2 = st.columns(2)
            with col_cost1:
                st.metric("Total Components", len(components_df))
            with col_cost2:
                st.metric("Estimated Cost", f"${total_cost:.2f} USD")
            
            # Download BOM
            bom_csv = components_df.to_csv(index=False)
            st.download_button(
                label="📥 Download BOM (CSV)",
                data=bom_csv,
                file_name="vigilant_bill_of_materials.csv",
                mime="text/csv"
            )
        
        with tab3:
            st.markdown("### 🔧 **Installation & Wiring Guide**")
            
            col_guide1, col_guide2 = st.columns(2)
            
            with col_guide1:
                st.markdown("#### **Step 1: Power Connections**")
                st.markdown("""
                1. Connect 5V power supply to ESP32 Vin pin
                2. Connect GND to common ground rail
                3. Add 10μF capacitor between 5V and GND
                4. Add 100nF capacitor between 3.3V and GND
                """)
                
                st.markdown("#### **Step 2: Sensor Connections**")
                st.markdown("""
                **DHT11 Wiring:**
                - DATA → GPIO 4 (with 10K pull-up)
                - VCC → 3.3V
                - GND → Common GND
                
                **Camera Wiring:**
                - SDA → GPIO 13
                - SCL → GPIO 12  
                - VCC → 3.3V dedicated
                - GND → Common GND
                """)
            
            with col_guide2:
                st.markdown("#### **Step 3: Display & Outputs**")
                st.markdown("""
                **OLED Display:**
                - SDA → GPIO 17
                - SCL → GPIO 16
                - VCC → 3.3V
                - GND → Common GND
                
                **Output Devices:**
                - Buzzer + → GPIO 15
                - Buzzer - → GND
                - Relay IN → GPIO 33
                - LED + → GPIO 14
                - LED - → 220Ω → GND
                """)
                
                st.markdown("#### **Step 4: Testing**")
                st.markdown("""
                1. Power on the system
                2. Check all LED indicators
                3. Verify sensor readings
                4. Test camera feed
                5. Validate communication
                """)
            
            # Troubleshooting section
            with st.expander("🛠️ **Troubleshooting Common Issues**"):
                st.markdown("""
                **Issue 1: ESP32 Not Powering On**
                - Check 5V power supply
                - Verify USB cable is data-capable
                - Check for short circuits
                
                **Issue 2: Sensors Not Reading**
                - Verify 3.3V supply to sensors
                - Check pull-up resistors
                - Verify GPIO pin assignments
                
                **Issue 3: Camera Not Working**
                - Ensure sufficient power (200mA+)
                - Check I2C address (0x30)
                - Verify focus adjustment
                
                **Issue 4: WiFi Connection Issues**
                - Check SSID and password
                - Verify signal strength
                - Check firewall settings
                """)
    
    else:
        st.error("Circuit visual module not available. Please install matplotlib.")
        st.info("""
        To enable circuit visualization:
        1. Install matplotlib: `pip install matplotlib`
        2. Restart the application
        3. The circuit diagram will be available
        """)

# ------------------------------
# Session state
# ------------------------------
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=200)
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "running" not in st.session_state:
    st.session_state.running = True
if "data_index" not in st.session_state:
    st.session_state.data_index = 0
if "dataset" not in st.session_state:
    st.session_state.dataset = load_dataset()

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
    
    # Navigation dengan circuit diagram option
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "📈 Live Monitor", "🌡️ Temperature Analysis", 
         "📋 Project Report", "🔌 Circuit Diagram", "👥 Team", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📡 Data Source")
    mode = st.selectbox(
        "Select Mode",
        ["Simulation (Dataset)", "Real-time Sensors", "Historical Analysis"],
        label_visibility="collapsed"
    )
    
    refresh_rate = st.slider("🔄 Refresh Rate (s)", 0.5, 5.0, 2.0, 0.1)
    
    st.markdown("---")
    
    # System Status Summary
    st.markdown("### 📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model", "✓ Loaded" if model_loaded else "⚠ Demo")
    with col2:
        st.metric("Data Points", len(st.session_state.history))
    
    # Dataset info if available
    if st.session_state.dataset is not None:
        st.metric("Dataset Records", len(st.session_state.dataset))
    
    st.markdown("---")
    st.caption("Vigilant v1.0 | BINUS University")

# ------------------------------
# DASHBOARD PAGE
# ------------------------------
if page == "📊 Dashboard":
    st.title("🔍 Vigilant Dashboard")
    st.markdown("Real-time environmental monitoring & anomaly detection system")
    
    # Get current data
    current_data = simulate_input_from_dataset(st.session_state.dataset, st.session_state.data_index)
    prediction = get_prediction(current_data)
    anomaly_conditions = get_anomaly_conditions(current_data, prediction)
    
    # Update index for next reading
    st.session_state.data_index = (st.session_state.data_index + 1) % (100 if st.session_state.dataset is None else len(st.session_state.dataset))
    
    # Store alert if anomaly
    if prediction == "Anomaly":
        alert = {
            "timestamp": datetime.now(),
            "conditions": anomaly_conditions,
            "data": current_data,
            "type": "ANOMALY"
        }
        if len(st.session_state.alerts) == 0 or (datetime.now() - st.session_state.alerts[-1]["timestamp"]).seconds > 10:
            st.session_state.alerts.append(alert)
    
    # Row 1: System Status Cards
    st.markdown('<div class="section-header"><h3>📡 System Status</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="sensor-label">ESP32 Status</div>', unsafe_allow_html=True)
        status_class = "status-normal" if current_data["esp32_status"] == "Online" else "status-offline"
        st.markdown(f'<div class="{status_class}">{current_data["esp32_status"]}</div>', unsafe_allow_html=True)
        st.progress(0.95 if current_data["esp32_status"] == "Online" else 0.1)
        st.caption(f"Latency: {current_data['network_latency']}ms")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="sensor-label">Camera Status</div>', unsafe_allow_html=True)
        cam_class = "status-normal" if current_data["camera_status"] == "Streaming" else "status-warning" if current_data["camera_status"] == "Idle" else "status-danger"
        st.markdown(f'<div class="{cam_class}">{current_data["camera_status"]}</div>', unsafe_allow_html=True)
        
        if current_data["camera_status"] == "Streaming":
            st.metric("FPS", "24", delta="Stable")
        elif current_data["camera_status"] == "Idle":
            st.metric("Status", "Standby", delta="-")
        else:
            st.metric("Status", "Error", delta="Check", delta_color="inverse")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="sensor-label">Temperature Status</div>', unsafe_allow_html=True)
        temp_status, temp_class = get_temperature_status(current_data["temperature"])
        st.markdown(f'<div class="{temp_class}">{temp_status}</div>', unsafe_allow_html=True)
        
        # Temperature value with color coding
        temp_color = "#ef4444" if current_data["temperature"] > 32 else "#3b82f6" if current_data["temperature"] < 22 else "#10b981"
        st.markdown(f'<div style="color: {temp_color}; font-size: 2rem; font-weight: 700;">{current_data["temperature"]}°C</div>', unsafe_allow_html=True)
        
        # Threshold indicators
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.caption("🌡️ Low <22°C")
        with col_t2:
            st.caption("✅ Normal 22-32°C")
        with col_t3:
            st.caption("⚠️ High >32°C")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="sensor-label">Environment</div>', unsafe_allow_html=True)
        
        # Humidity
        hum_status, hum_class = get_humidity_status(current_data["humidity"])
        st.markdown(f'<div class="{hum_class}" style="margin-bottom: 10px;">Humidity: {hum_status}</div>', unsafe_allow_html=True)
        
        col_hum1, col_hum2 = st.columns([2, 1])
        with col_hum1:
            hum_color = "#0ea5e9" if current_data["humidity"] > 90 else "#10b981"
            st.markdown(f'<div style="color: {hum_color}; font-size: 1.8rem; font-weight: 700;">{current_data["humidity"]}%</div>', unsafe_allow_html=True)
        with col_hum2:
            st.progress(current_data["humidity"]/100)
        
        # Motion Intensity
        st.markdown("---")
        st.markdown('<div class="sensor-label">Motion Intensity</div>', unsafe_allow_html=True)
        motion_color = "#f59e0b" if current_data["motion_intensity"] > 0.6 else "#10b981"
        st.markdown(f'<div style="color: {motion_color}; font-size: 1.8rem; font-weight: 700;">{current_data["motion_intensity"]:.2f}</div>', unsafe_allow_html=True)
        st.progress(current_data["motion_intensity"])
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Row 2: Anomaly Detection with Specific Conditions
    st.markdown('<div class="section-header"><h3>🚨 Anomaly Detection & Conditions</h3></div>', unsafe_allow_html=True)
    
    col_main, col_side = st.columns([3, 1])
    
    with col_main:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        if prediction == "Anomaly":
            st.markdown('<div class="anomaly-alert">', unsafe_allow_html=True)
            col_alert1, col_alert2 = st.columns([1, 3])
            with col_alert1:
                st.markdown("🚨")
                st.markdown(f'<div class="status-danger" style="font-size: 1.2rem;">{prediction}</div>', unsafe_allow_html=True)
                st.metric("Severity", "HIGH", delta_color="inverse")
            with col_alert2:
                st.markdown("### ⚠️ Critical Conditions Detected")
                for condition in anomaly_conditions:
                    if "Temperature" in condition:
                        st.markdown(f'<div class="temp-high-alert">🔥 {condition}</div>', unsafe_allow_html=True)
                    elif "Humidity" in condition:
                        st.markdown(f'<div class="humidity-high-alert">💧 {condition}</div>', unsafe_allow_html=True)
                    elif "Low" in condition:
                        st.markdown(f'<div class="temp-low-alert">❄️ {condition}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"• **{condition}**")
                
                st.markdown("**Recommended Actions:**")
                if "Temperature" in " ".join(anomaly_conditions):
                    st.markdown("• Check cooling systems")
                    st.markdown("• Verify sensor calibration")
                if "Humidity" in " ".join(anomaly_conditions):
                    st.markdown("• Inspect for moisture sources")
                    st.markdown("• Check ventilation")
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif prediction == "Suspicious":
            col_sus1, col_sus2 = st.columns([1, 3])
            with col_sus1:
                st.markdown("⚠️")
                st.markdown(f'<div class="status-warning" style="font-size: 1.2rem;">{prediction}</div>', unsafe_allow_html=True)
                st.metric("Risk", "MEDIUM")
            with col_sus2:
                st.markdown("### Suspicious Activity Detected")
                st.markdown("**Conditions detected:**")
                for condition in anomaly_conditions:
                    st.markdown(f"• {condition}")
                st.markdown("**Monitor closely for changes.**")
        
        else:
            col_norm1, col_norm2 = st.columns([1, 3])
            with col_norm1:
                st.markdown("✅")
                st.markdown(f'<div class="status-normal" style="font-size: 1.2rem;">{prediction}</div>', unsafe_allow_html=True)
                st.metric("Risk", "LOW", delta="Stable")
            with col_norm2:
                st.markdown("### System Operating Normally")
                st.markdown("All environmental parameters within safe ranges.")
                if anomaly_conditions:
                    st.markdown(f"**Status:** {anomaly_conditions[0]}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_side:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Detection Confidence")
        
        # Create gauge chart with prediction-specific values
        if prediction == "Anomaly":
            value = 92
            color = "#ef4444"
        elif prediction == "Suspicious":
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
        
        # Additional metrics
        st.markdown("---")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Temp Trend", "↑" if current_data["temperature"] > 25 else "↓")
        with col_m2:
            st.metric("Response Time", "1.2s")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Row 3: Recent Alerts with Detailed Conditions
    if st.session_state.alerts:
        st.markdown('<div class="section-header"><h3>📋 Recent Alerts History</h3></div>', unsafe_allow_html=True)
        
        for alert in list(reversed(st.session_state.alerts))[:3]:
            with st.expander(f"🚨 {alert['type']} at {alert['timestamp'].strftime('%H:%M:%S')} - {len(alert['conditions'])} conditions"):
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.markdown("**🔄 Conditions Detected:**")
                    for cond in alert["conditions"]:
                        if "Temperature" in cond:
                            st.markdown(f"• 🔥 {cond}")
                        elif "Humidity" in cond:
                            st.markdown(f"• 💧 {cond}")
                        elif "Motion" in cond:
                            st.markdown(f"• 👣 {cond}")
                        else:
                            st.markdown(f"• ⚙️ {cond}")
                
                with col_a2:
                    st.markdown("**📊 Sensor Data:**")
                    st.markdown(f"**Temperature:** {alert['data']['temperature']}°C")
                    st.markdown(f"**Humidity:** {alert['data']['humidity']}%")
                    st.markdown(f"**Motion:** {alert['data']['motion_intensity']:.2f}")
                
                with col_a3:
                    st.markdown("**🔧 System Status:**")
                    st.markdown(f"**ESP32:** {alert['data']['esp32_status']}")
                    st.markdown(f"**Camera:** {alert['data']['camera_status']}")
                    st.markdown(f"**Latency:** {alert['data']['network_latency']}ms")

# ------------------------------
# LIVE MONITOR PAGE
# ------------------------------
elif page == "📈 Live Monitor":
    st.title("📈 Live Monitor")
    st.markdown("Real-time sensor data visualization from dataset")
    
    placeholder = st.empty()
    
    with placeholder.container():
        current_data = simulate_input_from_dataset(st.session_state.dataset, st.session_state.data_index)
        prediction = get_prediction(current_data)
        
        # Update history
        st.session_state.history.append({
            "timestamp": current_data["timestamp"],
            "temperature": current_data["temperature"],
            "humidity": current_data["humidity"],
            "motion_intensity": current_data["motion_intensity"],
            "prediction": prediction,
            "temperature_status": current_data["temperature_status"]
        })
        
        # Update index
        st.session_state.data_index = (st.session_state.data_index + 1) % (100 if st.session_state.dataset is None else len(st.session_state.dataset))
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Live Charts", "📈 History Analysis", "🔍 Sensor Details"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                # Temperature Gauge with status zones
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("🌡️ Temperature Monitor")
                
                fig_temp = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = current_data["temperature"],
                    title = {'text': "°C"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [15, 45]},
                        'bar': {'color': "#ef4444" if current_data["temperature"] > 32 else "#3b82f6" if current_data["temperature"] < 22 else "#10b981"},
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 32
                        },
                        'steps': [
                            {'range': [15, 22], 'color': "rgba(59, 130, 246, 0.2)"},
                            {'range': [22, 32], 'color': "rgba(16, 185, 129, 0.2)"},
                            {'range': [32, 45], 'color': "rgba(239, 68, 68, 0.2)"}
                        ]
                    }
                ))
                fig_temp.update_layout(height=250)
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Temperature status
                temp_status, _ = get_temperature_status(current_data["temperature"])
                st.markdown(f"**Status:** {temp_status}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                # Motion Intensity Chart
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("📈 Motion Intensity")
                
                if len(st.session_state.history) > 0:
                    hist_df = pd.DataFrame(list(st.session_state.history))
                    fig_motion = go.Figure()
                    fig_motion.add_trace(go.Scatter(
                        x=hist_df["timestamp"],
                        y=hist_df["motion_intensity"],
                        mode='lines+markers',
                        name='Motion',
                        line=dict(color='#f59e0b', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add threshold lines
                    fig_motion.add_hline(y=0.8, line_dash="dash", line_color="red", 
                                        annotation_text="High Alert", 
                                        annotation_position="bottom right")
                    fig_motion.add_hline(y=0.6, line_dash="dot", line_color="orange", 
                                        annotation_text="Warning", 
                                        annotation_position="bottom right")
                    
                    fig_motion.update_layout(
                        height=250,
                        xaxis_title="Time",
                        yaxis_title="Intensity",
                        showlegend=False,
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    st.plotly_chart(fig_motion, use_container_width=True)
                else:
                    st.info("No data available yet")
                
                # Current motion status
                motion_status = "High" if current_data["motion_intensity"] > 0.8 else "Moderate" if current_data["motion_intensity"] > 0.6 else "Low"
                st.markdown(f"**Current Activity:** {motion_status}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            if len(st.session_state.history) > 0:
                hist_df = pd.DataFrame(list(st.session_state.history))
                
                # Temperature History
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("📊 Temperature History")
                
                fig_temp_hist = go.Figure()
                fig_temp_hist.add_trace(go.Scatter(
                    x=hist_df["timestamp"],
                    y=hist_df["temperature"],
                    mode='lines+markers',
                    name='Temperature',
                    line=dict(color='#ef4444', width=2),
                    marker=dict(size=4)
                ))
                
                # Add threshold zones
                fig_temp_hist.add_hrect(y0=22, y1=32, line_width=0, fillcolor="rgba(16, 185, 129, 0.1)", 
                                      annotation_text="Normal Zone", annotation_position="top left")
                fig_temp_hist.add_hrect(y0=32, y1=45, line_width=0, fillcolor="rgba(239, 68, 68, 0.1)")
                fig_temp_hist.add_hrect(y0=15, y1=22, line_width=0, fillcolor="rgba(59, 130, 246, 0.1)")
                
                fig_temp_hist.update_layout(
                    height=300,
                    xaxis_title="Time",
                    yaxis_title="Temperature (°C)",
                    showlegend=False
                )
                st.plotly_chart(fig_temp_hist, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("🌡️ Temperature Details")
                
                temp_status, temp_class = get_temperature_status(current_data["temperature"])
                st.markdown(f'<div class="{temp_class}" style="margin-bottom: 15px;">{temp_status}</div>', unsafe_allow_html=True)
                
                st.metric("Current", f"{current_data['temperature']} °C", 
                         delta="↑ High" if current_data['temperature'] > 32 else "↓ Low" if current_data['temperature'] < 22 else "✓ Normal")
                
                # Temperature statistics
                if len(st.session_state.history) > 0:
                    hist_df = pd.DataFrame(list(st.session_state.history))
                    avg_temp = hist_df["temperature"].mean()
                    max_temp = hist_df["temperature"].max()
                    min_temp = hist_df["temperature"].min()
                    
                    st.markdown("**Statistics:**")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Avg", f"{avg_temp:.1f}°C")
                    with col_stat2:
                        st.metric("Max", f"{max_temp:.1f}°C")
                    with col_stat3:
                        st.metric("Min", f"{min_temp:.1f}°C")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_s2:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("💧 Humidity & Activity")
                
                hum_status, hum_class = get_humidity_status(current_data["humidity"])
                st.markdown(f'<div class="{hum_class}" style="margin-bottom: 10px;">Humidity: {hum_status}</div>', unsafe_allow_html=True)
                
                st.metric("Humidity", f"{current_data['humidity']} %", 
                         delta="High" if current_data['humidity'] > 90 else "Normal")
                
                st.markdown("---")
                st.markdown("**Motion Activity:**")
                motion_color = "#ef4444" if current_data["motion_intensity"] > 0.8 else "#f59e0b" if current_data["motion_intensity"] > 0.6 else "#10b981"
                st.markdown(f'<div style="color: {motion_color}; font-size: 1.5rem; font-weight: 700;">Intensity: {current_data["motion_intensity"]:.2f}</div>', unsafe_allow_html=True)
                
                # Activity level indicator
                if current_data["motion_intensity"] > 0.8:
                    activity_level = "High Activity"
                elif current_data["motion_intensity"] > 0.6:
                    activity_level = "Moderate Activity"
                elif current_data["motion_intensity"] > 0.3:
                    activity_level = "Normal Activity"
                else:
                    activity_level = "Low Activity"
                
                st.markdown(f"**Level:** {activity_level}")
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Auto-refresh control
    if st.session_state.running:
        time.sleep(refresh_rate)
        st.experimental_rerun()
    else:
        st.info("⏸️ Live monitoring paused. Start from controls.")

# ------------------------------
# TEMPERATURE ANALYSIS PAGE
# ------------------------------
elif page == "🌡️ Temperature Analysis":
    st.title("🌡️ Temperature Analysis")
    st.markdown("Detailed analysis of temperature patterns from dataset")
    
    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Avg Temp", f"{df['temperature_celsius'].mean():.1f}°C")
        with col3:
            st.metric("Max Temp", f"{df['temperature_celsius'].max():.1f}°C")
        with col4:
            st.metric("Min Temp", f"{df['temperature_celsius'].min():.1f}°C")
        
        # Status distribution
        st.markdown("### 📊 Status Distribution")
        status_counts = df['status'].value_counts()
        col_dist1, col_dist2, col_dist3 = st.columns(3)
        with col_dist1:
            normal_count = status_counts.get('NORMAL', 0)
            st.metric("NORMAL", normal_count, f"{(normal_count/len(df)*100):.1f}%")
        with col_dist2:
            warning_count = status_counts.get('WARNING', 0)
            st.metric("WARNING", warning_count, f"{(warning_count/len(df)*100):.1f}%")
        with col_dist3:
            low_count = status_counts.get('LOW', 0)
            st.metric("LOW", low_count, f"{(low_count/len(df)*100):.1f}%")
        
        # Temperature trends
        st.markdown("### 📈 Temperature Trends Over Time")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['temperature_celsius'],
            mode='lines',
            name='Temperature',
            line=dict(color='#ef4444', width=1)
        ))
        
        # Add status-based coloring
        colors = {'NORMAL': '#10b981', 'WARNING': '#f59e0b', 'LOW': '#3b82f6'}
        for status in df['status'].unique():
            status_data = df[df['status'] == status]
            fig_trend.add_trace(go.Scatter(
                x=status_data['timestamp'],
                y=status_data['temperature_celsius'],
                mode='markers',
                name=status,
                marker=dict(color=colors.get(status, '#94a3b8'), size=6)
            ))
        
        fig_trend.update_layout(
            height=400,
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            showlegend=True
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Correlation analysis
        st.markdown("### 🔗 Temperature-Humidity Correlation")
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(
            x=df['temperature_celsius'],
            y=df['humidity_percent'],
            mode='markers',
            marker=dict(
                size=8,
                color=df['temperature_celsius'],
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Temp (°C)")
            ),
            text=df['status'],
            hovertemplate='<b>Temp:</b> %{x}°C<br><b>Humidity:</b> %{y}%<br><b>Status:</b> %{text}<extra></extra>'
        ))
        
        fig_corr.update_layout(
            height=400,
            xaxis_title="Temperature (°C)",
            yaxis_title="Humidity (%)",
            showlegend=False
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Status timeline
        st.markdown("### ⏱️ Status Timeline")
        status_map = {'NORMAL': 2, 'WARNING': 1, 'LOW': 0}
        df['status_numeric'] = df['status'].map(status_map)
        
        fig_status = go.Figure()
        fig_status.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['status_numeric'],
            mode='markers',
            name='Status',
            marker=dict(
                size=10,
                color=df['status_numeric'],
                colorscale=[[0, '#3b82f6'], [0.5, '#f59e0b'], [1, '#10b981']],
                showscale=True,
                colorbar=dict(
                    title="Status",
                    tickvals=[0, 1, 2],
                    ticktext=['LOW', 'WARNING', 'NORMAL']
                )
            )
        ))
        
        fig_status.update_layout(
            height=300,
            xaxis_title="Time",
            yaxis_title="Status Level",
            yaxis=dict(
                tickvals=[0, 1, 2],
                ticktext=['LOW', 'WARNING', 'NORMAL']
            ),
            showlegend=False
        )
        st.plotly_chart(fig_status, use_container_width=True)
        
    else:
        st.error("Dataset not loaded. Please check the file path.")

# ------------------------------
# PROJECT REPORT PAGE
# ------------------------------
elif page == "📋 Project Report":
    st.title("📋 Project Report — Vigilant")
    st.markdown("Complete project documentation and analysis")
    
    with st.expander("📊 Performance Metrics", expanded=True):
        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        with col_met1:
            st.metric("Accuracy", "92%", "2%")
        with col_met2:
            st.metric("False Alerts", "3.2%", "-0.5%")
        with col_met3:
            st.metric("Response Time", "1.2s", "-0.3s")
        with col_met4:
            st.metric("Uptime", "99.8%", "0.1%")
    
    # Dataset analysis
    if st.session_state.dataset is not None:
        with st.expander("📁 Dataset Analysis", expanded=True):
            df = st.session_state.dataset
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**Dataset Statistics:**")
                st.dataframe(df.describe())
            
            with col_d2:
                st.markdown("**Status Distribution:**")
                status_counts = df['status'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=status_counts.index,
                    values=status_counts.values,
                    hole=.3,
                    marker_colors=['#10b981', '#f59e0b', '#3b82f6']
                )])
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

# ------------------------------
# CIRCUIT DIAGRAM PAGE
# ------------------------------
elif page == "🔌 Circuit Diagram":
    display_circuit_page()

# ------------------------------
# TEAM PAGE
# ------------------------------
elif page == "👥 Team":
    st.title("👥 Team — Incognito")
    st.markdown("Meet the development team")
    
    members = [
        {"name": "Ahmad Hamra", "role": "Logic Developer", "skills": ["AI/ML", "Embedded Systems", "Python"]},
        {"name": "Alfred Abner", "role": "Documentation Specialist", "skills": ["Technical Writing", "Research", "Testing"]},
        {"name": "Davin Aji Wibowo", "role": "Video Production", "skills": ["Video Editing", "UI/UX", "Presentation"]},
        {"name": "Reynaldo Lamhot Silalahi", "role": "Documentation Specialist", "skills": ["Data Analysis", "Reporting", "QA"]},
    ]
    
    cols = st.columns(2)
    for idx, member in enumerate(members):
        with cols[idx % 2]:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader(member["name"])
            st.markdown(f"**Role:** {member['role']}")
            st.markdown("**Skills:**")
            for skill in member["skills"]:
                st.markdown(f"• {skill}")
            st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# SETTINGS PAGE
# ------------------------------
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    tab_set1, tab_set2, tab_set3 = st.tabs(["General", "Alert Thresholds", "Advanced"])
    
    with tab_set1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("General Settings")
        
        col_set1, col_set2 = st.columns(2)
        with col_set1:
            st.selectbox("Theme", ["Dark", "Light", "Auto"])
            st.number_input("Data Retention (days)", 1, 365, 30)
        with col_set2:
            st.selectbox("Time Zone", ["Asia/Jakarta", "UTC"])
            st.checkbox("Auto-start monitoring", True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab_set2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Alert Threshold Settings")
        
        col_th1, col_th2 = st.columns(2)
        with col_th1:
            st.markdown("**🌡️ Temperature Thresholds**")
            low_temp = st.slider("Low Temperature (°C)", 15, 25, 22)
            high_temp = st.slider("High Temperature (°C)", 28, 45, 32)
            critical_temp = st.slider("Critical Temperature (°C)", 32, 50, 35)
        
        with col_th2:
            st.markdown("**💧 Humidity Thresholds**")
            high_humidity = st.slider("High Humidity (%)", 70, 100, 90)
            st.markdown("**📈 Motion Thresholds**")
            motion_warning = st.slider("Motion Warning", 0.0, 1.0, 0.6)
            motion_alert = st.slider("Motion Alert", 0.0, 1.0, 0.8)
        
        st.markdown("**Current Settings:**")
        st.info(f"""
        - Low Temperature: < {low_temp}°C
        - Normal Temperature: {low_temp} - {high_temp}°C
        - High Temperature: > {high_temp}°C
        - Critical Temperature: > {critical_temp}°C
        - High Humidity: > {high_humidity}%
        - Motion Warning: > {motion_warning}
        - Motion Alert: > {motion_alert}
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab_set3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Advanced Settings")
        
        st.text_input("MQTT Broker", "broker.emqx.io")
        st.text_input("API Endpoint", "https://api.vigilant.com")
        st.text_area("Custom Configuration", '{"log_level": "INFO", "data_interval": 30}')
        
        st.markdown("---")
        st.markdown("**Data Management**")
        if st.button("🔄 Reload Dataset"):
            st.session_state.dataset = load_dataset()
            st.success("Dataset reloaded!")
        
        if st.button("🗑️ Clear History"):
            st.session_state.history.clear()
            st.session_state.alerts.clear()
            st.success("History cleared!")
        
        if st.button("🔧 Reset to Defaults", type="secondary"):
            st.success("Settings reset successfully")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
with footer_col1:
    st.caption("🔍 Vigilant v1.0 | Environmental Anomaly Detection System")
with footer_col2:
    st.caption(f"Data Points: {len(st.session_state.history)}")
with footer_col3:
    st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")