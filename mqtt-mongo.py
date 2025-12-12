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
person_alerts_collection = db["person_alerts"]

# Store tracking for persons
person_tracking = {}  # Track last time each person was seen

# MQTT callback
def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        print(f"[{datetime.now()}] Received: {data}")
        
        # Add timestamp
        data["timestamp"] = datetime.now()
        
        # Check if this is a person detection alert
        if "alert_type" in data and data["alert_type"] == "person_staying":
            # Handle person detection alert
            handle_person_alert(data)
        else:
            # Handle regular sensor data (just store, no anomaly checking)
            handle_sensor_data(data)
        
    except Exception as e:
        print(f"Error processing message: {e}")

def handle_sensor_data(data):
    """Handle regular sensor data - just store, no anomaly checking"""
    # Insert to MongoDB readings collection
    readings_collection.insert_one(data)
    print(f"✓ Sensor data inserted to readings collection")

def handle_person_alert(data):
    """Handle person detection alerts - CREATE ALERT WHEN PERSON STAYS FOR 5 SECONDS"""
    try:
        person_id = data.get("person_id")
        duration = data.get("duration_seconds", 0)
        
        # Store in person alerts collection
        person_alerts_collection.insert_one(data)
        print(f"✓ Person alert stored: Person {person_id} stayed for {duration}s at {data.get('position')}")
        
        # Check if this is exactly 5 seconds (person just left after 5 seconds)
        if duration == 5.0:
            # CREATE ALERT for person detection (exactly 5 seconds)
            alert_data = {
                "timestamp": data["timestamp"],
                "type": "PERSON_DETECTED",
                "person_id": person_id,
                "duration_seconds": duration,
                "position": data.get("position"),
                "message": f"Person {person_id} detected for exactly {duration} seconds",
                "severity": "MEDIUM",
                "source": "Person Detection System",
                "data": data
            }
            
            # Insert alert
            alerts_collection.insert_one(alert_data)
            print(f"⚠️ ALERT: Person {person_id} detected and left after {duration} seconds")
            
        else:
            print(f"ℹ️ Person {person_id} stayed for {duration}s (not 5s, no alert)")
                
    except Exception as e:
        print(f"Error handling person alert: {e}")

# MQTT setup
client = mqtt.Client()
client.on_message = on_message

try:
    client.connect("broker.emqx.io", 1883, 60)
    
    # Subscribe to both topics
    client.subscribe("sic7/teamincognito/sensors")
    client.subscribe("sic7/teamincognito/alarm")
    
    print("=" * 60)
    print("🚀 MQTT to MongoDB Bridge Started")
    print("=" * 60)
    print("📡 Listening for MQTT messages on topics:")
    print("   • sic7/teamincognito/sensors")
    print("   • sic7/teamincognito/person_alerts")
    print("=" * 60)
    print("📊 Person Detection Logic:")
    print("   • Alert created ONLY when person stays for exactly 5 seconds")
    print("   • 5 seconds = person detected and then left")
    print("=" * 60)
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    client.loop_forever()
    
except KeyboardInterrupt:
    print("\n🛑 Stopping MQTT client...")
    client.disconnect()
    print("✅ MQTT client stopped")
except Exception as e:
    print(f"❌ Error: {e}")