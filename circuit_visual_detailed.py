# circuit_visual_detailed.py
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from PIL import Image
import io

def create_detailed_circuit():
    """Membuat gambar schematic yang lebih detail"""
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Warna tema konsisten dengan UI
    colors = {
        'esp32': '#f59e0b',      # Orange/Warning
        'dht11': '#3b82f6',      # Blue
        'camera': '#ef4444',     # Red/Danger
        'oled': '#10b981',       # Green/Success
        'buzzer': '#8b5cf6',     # Purple
        'relay': '#ec4899',      # Pink
        'button': '#64748b',     # Gray
        'power': '#fbbf24',      # Yellow
        'wifi': '#0ea5e9',       # Sky blue
        'led': '#22c55e',        # Green LED
        'resistor': '#a855f7',   # Purple resistor
    }
    
    # Define components dengan posisi yang lebih terstruktur
    components = {
        'power': {'x': 6, 'y': 11, 'w': 1.5, 'h': 0.8, 'label': '🔋 POWER\n5V DC'},
        'esp32': {'x': 6, 'y': 9, 'w': 3.0, 'h': 2.0, 'label': '⚡ ESP32\nDevKit V1\n240MHz Dual Core'},
        'wifi': {'x': 9, 'y': 9, 'w': 1.8, 'h': 1.0, 'label': '📶 WiFi\n802.11 b/g/n'},
        'dht11': {'x': 2, 'y': 7, 'w': 2.0, 'h': 1.2, 'label': '🌡️ DHT11\nTemperature\nHumidity'},
        'camera': {'x': 10, 'y': 7, 'w': 2.0, 'h': 1.2, 'label': '📷 CAMERA\nOV2640\n2MP'},
        'oled': {'x': 6, 'y': 5, 'w': 2.0, 'h': 1.2, 'label': '📺 OLED\nSSD1306\n128x64'},
        'buzzer': {'x': 3, 'y': 3, 'w': 1.8, 'h': 1.0, 'label': '🔊 BUZZER\nActive\n85dB'},
        'relay': {'x': 9, 'y': 3, 'w': 1.8, 'h': 1.0, 'label': '⚡ RELAY\n5V 10A\nOptocoupler'},
        'button': {'x': 6, 'y': 1.5, 'w': 1.5, 'h': 0.8, 'label': '🔘 BUTTON\nReset/Control'},
        'led': {'x': 1, 'y': 9, 'w': 1.2, 'h': 0.7, 'label': '💡 LED\nStatus'},
        'resistor': {'x': 1, 'y': 10.2, 'w': 1.0, 'h': 0.5, 'label': '📏 RES\n220Ω'},
    }
    
    # Draw components dengan styling yang lebih baik
    for name, pos in components.items():
        # Kotak komponen dengan rounded corners
        box = FancyBboxPatch(
            (pos['x'] - pos['w']/2, pos['y'] - pos['h']/2),
            pos['w'], pos['h'],
            boxstyle="round,pad=0.2,rounding_size=0.15",
            facecolor=colors.get(name, '#1e293b'),
            edgecolor='white',
            linewidth=2,
            alpha=0.9,
            linestyle='-'
        )
        ax.add_patch(box)
        
        # Teks komponen dengan line break
        lines = pos['label'].split('\n')
        for i, line in enumerate(lines):
            y_offset = pos['h']/2 - 0.15 - i*0.2
            ax.text(pos['x'], pos['y'] - y_offset, 
                   line, 
                   ha='center', va='center',
                   color='white',
                   fontsize=7 if len(lines) > 2 else 8,
                   fontweight='bold' if i == 0 else 'normal',
                   linespacing=1.2)
    
    # Draw connections dengan label
    connections = [
        ('power', 'esp32', '5V Power', '#fbbf24', 3.0),
        ('esp32', 'wifi', 'SPI Bus', '#0ea5e9', 2.0),
        ('esp32', 'dht11', 'GPIO 4\nOne-Wire', '#3b82f6', 2.0),
        ('esp32', 'camera', 'I2C\nGPIO 12-13', '#ef4444', 2.0),
        ('esp32', 'oled', 'I2C\nGPIO 16-17', '#10b981', 2.0),
        ('esp32', 'buzzer', 'GPIO 15\nPWM', '#8b5cf6', 2.0),
        ('esp32', 'relay', 'GPIO 33\nControl', '#ec4899', 2.0),
        ('esp32', 'button', 'GPIO 32\nInput', '#64748b', 2.0),
        ('esp32', 'led', 'GPIO 14\nStatus', '#22c55e', 2.0),
        ('led', 'resistor', '220Ω\nCurrent Limit', '#a855f7', 1.5),
        ('resistor', 'power', '3.3V', '#fbbf24', 1.5),
    ]
    
    for start, end, label, color, width in connections:
        x1, y1 = components[start]['x'], components[start]['y'] - components[start]['h']/2
        x2, y2 = components[end]['x'], components[end]['y'] + components[end]['h']/2
        
        # Atur arah koneksi berdasarkan posisi
        if start == 'resistor' and end == 'power':
            x1, y1 = components[start]['x'], components[start]['y'] + components[start]['h']/2
            x2, y2 = components[end]['x'], components[end]['y'] - components[end]['h']/2
        elif end == 'led' or end == 'resistor':
            x2, y2 = components[end]['x'], components[end]['y'] - components[end]['h']/2
        
        # Garis koneksi dengan arrow
        line = ConnectionPatch(
            (x1, y1), (x2, y2),
            coordsA="data", coordsB="data",
            arrowstyle="->" if 'GPIO' in label else "-", 
            color=color,
            linewidth=width,
            alpha=0.8,
            connectionstyle="arc3,rad=0.2",
            shrinkA=5,
            shrinkB=5
        )
        ax.add_patch(line)
        
        # Label untuk koneksi
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2 + 0.3
        
        ax.text(mid_x, mid_y, label,
                ha='center', va='center',
                color=color,
                fontsize=7,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor='#1e293b', 
                         edgecolor=color,
                         alpha=0.9,
                         linewidth=1))
    
    # Add power and ground symbols
    # Power symbol (VCC)
    ax.text(6, 11.8, '⏚',  # Ground symbol
           ha='center', va='center',
           color='#fbbf24',
           fontsize=20)
    
    ax.text(6, 11.6, 'GND',
           ha='center', va='center',
           color='#94a3b8',
           fontsize=8)
    
    # Add title dan subtitle
    ax.text(6, 12.2, 'VIGILANT SYSTEM - ELECTRONIC CIRCUIT DIAGRAM', 
           ha='center', va='center',
           color='white',
           fontsize=20,
           fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor='#1e293b', 
                    edgecolor='#0ea5e9',
                    alpha=0.9))
    
    ax.text(6, 11.9, 'Motion Anomaly Detection & Environmental Monitoring System', 
           ha='center', va='center',
           color='#94a3b8',
           fontsize=12)
    
    # Add legend box
    legend_elements = [
        patches.Patch(facecolor='#f59e0b', edgecolor='white', label='ESP32 Microcontroller'),
        patches.Patch(facecolor='#3b82f6', edgecolor='white', label='DHT11 Sensor'),
        patches.Patch(facecolor='#ef4444', edgecolor='white', label='Camera Module'),
        patches.Patch(facecolor='#10b981', edgecolor='white', label='OLED Display'),
    ]
    
    legend = ax.legend(handles=legend_elements, 
                      loc='lower left',
                      bbox_to_anchor=(0.02, 0.02),
                      fontsize=8,
                      facecolor='#1e293b',
                      edgecolor='#475569',
                      labelcolor='white')
    
    # Add grid background yang samar untuk alignment
    ax.grid(True, alpha=0.05, color='#475569', linestyle=':')
    
    plt.tight_layout()
    return fig

def fig_to_image(fig):
    """Convert matplotlib figure to PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf)
    return img

def display_detailed_circuit():
    """Menampilkan schematic detail di Streamlit"""
    
    st.set_page_config(layout="wide", page_title="Vigilant Circuit Diagram")
    
    st.title("🔌 Visual Circuit Schematic")
    st.markdown("### Interaktif Diagram Rangkaian Sistem Vigilant")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📐 Circuit Diagram", "🔧 Component Details", "📥 Download"])
    
    with tab1:
        # Buat dan tampilkan diagram
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = create_detailed_circuit()
            st.pyplot(fig)
            
            # Diagram notes
            st.markdown("""
            **Diagram Notes:**
            - Solid arrows menunjukkan arah komunikasi/data flow
            - Garis tanpa arrow menunjukkan power/supply connections
            - Warna berbeda menunjukkan tipe koneksi yang berbeda
            - Semua GPIO menggunakan level 3.3V logic
            """)
        
        with col2:
            st.markdown("### 🎨 **Color Legend**")
            
            color_legend = [
                ("🟡 ESP32", "f59e0b", "Microcontroller utama"),
                ("🔵 DHT11", "3b82f6", "Sensor suhu & kelembaban"),
                ("🔴 Camera", "ef4444", "ESP32-CAM module"),
                ("🟢 OLED", "10b981", "Display lokal 128x64"),
                ("🟣 Buzzer", "8b5cf6", "Audio alert 85dB"),
                ("🔘 Relay", "ec4899", "Power control 10A"),
                ("⚪ Button", "64748b", "Reset/control input"),
                ("💡 LED", "22c55e", "Status indicator"),
                ("📏 Resistor", "a855f7", "Current limiting"),
            ]
            
            for name, color, desc in color_legend:
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="width: 20px; height: 20px; background-color: #{color}; 
                         border-radius: 4px; margin-right: 10px; border: 1px solid white;"></div>
                    <div>
                        <strong>{name}</strong><br>
                        <small style="color: #94a3b8;">{desc}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 🔗 **Connection Types**")
            st.markdown("""
            - **Kuning (fbbf24):** Power supply (5V/3.3V)
            - **Biru (0ea5e9):** SPI communication
            - **Biru tua (3b82f6):** One-Wire protocol
            - **Merah (ef4444):** I2C Camera
            - **Hijau (10b981):** I2C Display
            - **Ungu (8b5cf6):** PWM output
            - **Pink (ec4899):** Digital output
            - **Abu (64748b):** Digital input
            """)
    
    with tab2:
        st.markdown("### 🔧 **Detail Komponen & Spesifikasi**")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("#### ⚡ **ESP32 DevKit V1**")
            st.markdown("""
            **Spesifikasi:**
            - Microcontroller: ESP32-D0WDQ6
            - CPU: Xtensa dual-core 32-bit LX6
            - Clock Speed: 240 MHz
            - SRAM: 520 KB
            - Flash: 4 MB
            - WiFi: 802.11 b/g/n
            - Bluetooth: 4.2 BR/EDR/BLE
            
            **Fungsi:** 
            - Kontroler utama sistem
            - Data processing
            - Wireless communication
            """)
            
            st.markdown("#### 🌡️ **DHT11 Sensor**")
            st.markdown("""
            **Spesifikasi:**
            - Temperature Range: 0-50°C ±2°C
            - Humidity Range: 20-90% ±5%
            - Sampling Rate: 1 Hz
            - Interface: One-Wire digital
            - Operating Voltage: 3.3V-5V
            
            **Fungsi:**
            - Monitoring suhu ruangan
            - Monitoring kelembaban
            - Data untuk anomaly detection
            """)
        
        with col_b:
            st.markdown("#### 📷 **ESP32-CAM Module**")
            st.markdown("""
            **Spesifikasi:**
            - Camera Sensor: OV2640
            - Resolution: 2 Megapixel
            - Max Image: 1600×1200
            - Format: JPEG, BMP, Grayscale
            - Interface: I2C + DVP
            - Lens: Focal Length 3.6mm
            
            **Fungsi:**
            - Video streaming real-time
            - Motion detection
            - Image capture untuk evidence
            """)
            
            st.markdown("#### 📺 **OLED SSD1306**")
            st.markdown("""
            **Spesifikasi:**
            - Size: 0.96 inch
            - Resolution: 128×64 pixels
            - Interface: I2C (0x3C)
            - Colors: Monochrome (White)
            - Viewing Angle: >160°
            - Power: 3.3V, 20mA max
            
            **Fungsi:**
            - Display status lokal
            - Sensor readings
            - System information
            """)
        
        with col_c:
            st.markdown("#### 🔊 **Active Buzzer**")
            st.markdown("""
            **Spesifikasi:**
            - Type: Active piezoelectric
            - Voltage: 3-5V DC
            - Sound Level: 85 dB @ 10cm
            - Frequency: 2300±300 Hz
            - Current: <30mA
            - Diameter: 12mm
            
            **Fungsi:**
            - Audio alert untuk anomaly
            - System notifications
            - Warning signals
            """)
            
            st.markdown("#### ⚡ **Relay Module**")
            st.markdown("""
            **Spesifikasi:**
            - Relay: 5V DC, 10A 250VAC
            - Optocoupler: PC817
            - Isolation: 1500V
            - Response Time: <10ms
            - Contact: Normally Open (NO)
            - Life: 100,000 operations
            
            **Fungsi:**
            - Kontrol perangkat eksternal
            - Emergency shutdown
            - Power management
            """)
    
    with tab3:
        st.markdown("### 📥 **Download Resources**")
        
        # Generate and provide download for circuit diagram
        fig = create_detailed_circuit()
        img = fig_to_image(fig)
        
        col_x, col_y = st.columns(2)
        
        with col_x:
            st.markdown("#### **Circuit Diagram**")
            st.image(img, caption="Vigilant Circuit Diagram", use_column_width=True)
            
            # Convert to bytes for download
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            st.download_button(
                label="📥 Download PNG Diagram",
                data=byte_im,
                file_name="vigilant_circuit_diagram.png",
                mime="image/png"
            )
        
        with col_y:
            st.markdown("#### **📋 Wiring Instructions**")
            st.markdown("""
            **Step-by-Step Wiring Guide:**
            
            1. **Power Connections:**
               - Connect 5V power to ESP32 Vin pin
               - Connect GND to common ground rail
            
            2. **Sensor Connections:**
               - DHT11 Data → GPIO 4
               - DHT11 VCC → 3.3V
               - DHT11 GND → GND
            
            3. **Display Connections:**
               - OLED SDA → GPIO 17
               - OLED SCL → GPIO 16
               - OLED VCC → 3.3V
               - OLED GND → GND
            
            4. **Camera Connections:**
               - CAM SDA → GPIO 13
               - CAM SCL → GPIO 12
               - CAM VCC → 3.3V
               - CAM GND → GND
            
            5. **Output Devices:**
               - Buzzer + → GPIO 15
               - Buzzer - → GND
               - Relay IN → GPIO 33
               - LED + → GPIO 14
               - LED - → 220Ω → GND
            
            6. **Input Device:**
               - Button → GPIO 32
               - Button → GND (pull-down)
            """)
            
            # Download wiring guide as text
            wiring_guide = """
VIGILANT SYSTEM - WIRING GUIDE
===============================

POWER SUPPLY:
- 5V DC to ESP32 Vin pin
- Common GND rail for all components

SENSOR WIRING (DHT11):
- DATA  → GPIO 4
- VCC   → 3.3V
- GND   → GND

DISPLAY WIRING (OLED SSD1306):
- SDA   → GPIO 17
- SCL   → GPIO 16
- VCC   → 3.3V
- GND   → GND

CAMERA WIRING (ESP32-CAM):
- SDA   → GPIO 13
- SCL   → GPIO 12
- VCC   → 3.3V
- GND   → GND

OUTPUT DEVICES:
- Buzzer (+) → GPIO 15
- Buzzer (-) → GND
- Relay IN   → GPIO 33
- LED (+)    → GPIO 14
- LED (-)    → 220Ω resistor → GND

INPUT DEVICE:
- Button     → GPIO 32
- Button     → GND (with 10K pull-down resistor)

NOTES:
- All logic levels: 3.3V
- Use proper current limiting resistors
- Ensure stable power supply
- Check connections before powering
            """
            
            st.download_button(
                label="📄 Download Wiring Guide",
                data=wiring_guide,
                file_name="vigilant_wiring_guide.txt",
                mime="text/plain"
            )
        
        # Additional resources
        st.markdown("---")
        st.markdown("#### **🔧 Additional Resources**")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            if st.button("📊 Generate Pinout Diagram"):
                st.info("Pinout diagram akan digenerate...")
                # Here you could add code to generate pinout diagram
        
        with col_r2:
            if st.button("🛒 Component Shopping List"):
                shopping_list = """
COMPONENT SHOPPING LIST - VIGILANT SYSTEM
==========================================

1. ESP32 DevKit V1             x1
2. DHT11 Temperature Sensor    x1  
3. ESP32-CAM Module           x1
4. OLED SSD1306 0.96"         x1
5. Active Buzzer 5V           x1
6. 5V Relay Module            x1
7. Push Button                x1
8. LED 5mm                    x1
9. Resistor 220Ω              x5
10. Resistor 10KΩ             x5
11. Breadboard                x1
12. Jumper Wires (M-M)        x30
13. Micro USB Cable           x1
14. 5V 2A Power Supply        x1

Estimated Cost: $40-60 USD
                """
                st.download_button(
                    label="🛒 Download Shopping List",
                    data=shopping_list,
                    file_name="vigilant_shopping_list.txt",
                    mime="text/plain"
                )
        
        with col_r3:
            if st.button("💡 Troubleshooting Guide"):
                st.info("Troubleshooting guide akan ditampilkan...")

def main():
    """Main function untuk menjalankan standalone"""
    display_detailed_circuit()

if __name__ == "__main__":
    main()