# mqtt_to_mongodb.py
import json
import paho.mqtt.client as mqtt
from pymongo import MongoClient
from datetime import datetime
import time

# MongoDB connection
mongo = MongoClient("mongodb+srv://incognito:incognito_sic7@incognito.andn28n.mongodb.net/?appName=Incognito")
db = mongo["sensorDB"]
readings_collection = db["readings"]
alerts_collection = db["alerts"]

# Store last alert time to prevent spam
last_alert_time = {}

# MQTT callback
def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        print(f"[{datetime.now()}] Received: {data}")
        
        # Add timestamp
        data["timestamp"] = datetime.now()
        
        # Insert to MongoDB readings collection
        readings_collection.insert_one(data)
        print(f"Inserted to readings collection")
        
        # Check for anomalies and create alert if needed
        check_anomalies(data)
        
    except Exception as e:
        print(f"Error processing message: {e}")

def check_anomalies(data):
    """Check sensor data for anomalies"""
    alerts = []
    
    # Temperature anomalies
    if "temperature" in data and data["temperature"] > 35:
        alerts.append(f"Critical High Temperature: {data['temperature']}°C")
    elif "temperature" in data and data["temperature"] > 32:
        alerts.append(f"High Temperature: {data['temperature']}°C")
    elif "temperature" in data and data["temperature"] < 22:
        alerts.append(f"Low Temperature: {data['temperature']}°C")
    
    # Humidity anomalies
    if "humidity" in data and data["humidity"] > 90:
        alerts.append(f"High Humidity: {data['humidity']}%")
    
    # Light anomalies
    if "light" in data and data["light"] > 900:
        alerts.append(f"Extreme Brightness: {data['light']} lux")
    elif "light" in data and data["light"] < 100:
        alerts.append(f"Extreme Darkness: {data['light']} lux")
    
    # Create alert if any anomalies found
    if alerts:
        # Prevent alert spam (max 1 per minute per type)
        alert_type = "sensor_anomaly"
        current_time = time.time()
        
        if alert_type not in last_alert_time or (current_time - last_alert_time[alert_type] > 60):
            alert_data = {
                "timestamp": data["timestamp"],
                "type": "SENSOR_ANOMALY",
                "conditions": alerts,
                "data": data,
                "severity": "HIGH" if "Critical" in " ".join(alerts) else "MEDIUM"
            }
            
            alerts_collection.insert_one(alert_data)
            last_alert_time[alert_type] = current_time
            print(f"Alert created: {alerts}")

# MQTT setup
client = mqtt.Client()
client.on_message = on_message

try:
    client.connect("broker.emqx.io", 1883, 60)
    client.subscribe("sic7/teamincognito/sensors")
    
    print("Listening for MQTT messages on topic 'sic7/teamincognito/sensors'...")
    print("Press Ctrl+C to stop")
    
    client.loop_forever()
    
except KeyboardInterrupt:
    print("\nStopping MQTT client...")
    client.disconnect()
except Exception as e:
    print(f"Error: {e}")