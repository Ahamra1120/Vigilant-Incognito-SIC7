#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>
#include <time.h>

// ====== KONFIGURASI WIFI & MQTT ======
const char* ssid = "Incognito";
const char* password = "12345678";

const char* mqtt_server = "broker.emqx.io";
const int mqtt_port = 1883;
const char* mqtt_topic = "sic7/teamincognito/sensors";

String clientID = "sic7_device_1" + String(random(0xffff), HEX);

// ====== KONFIGURASI OLED ======
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ====== SENSOR ======
#define DHTPIN 15
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

#define LDR_PIN 34   // ADC1 (lebih stabil)
int ldrValue = 0;
int lightPercent = 0;

// ====== LED ALERT ======
#define LED_PIN 26   // D26
bool ledState = LOW;
unsigned long lastLedToggle = 0;
const unsigned long ledBlinkInterval = 150; // ms

#define BUZZER_PIN 27
bool buzzerState = LOW;

// ====== VAR GLOBAL ======
WiFiClient espClient;
PubSubClient client(espClient);

float temperature = 0; float humidity = 0;
unsigned long lastMqttPublish = 0;
const unsigned long mqttPublishInterval = 5000; // 5s

// Alert state
volatile bool alertActive = false;
String alertSource = ""; // store message if needed

// ====== TIME (GMT+7) ======
const long gmtOffset_sec = 7 * 3600;
const int daylightOffset_sec = 0;

String getTimestamp() {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    return String("Time Err");
  }
  char buf[25];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &timeinfo);
  return String(buf);
}

void drawHeader(const char *subtitle) {
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(2, 0);
  display.println(subtitle);
  display.drawFastHLine(0, 10, SCREEN_WIDTH, SSD1306_WHITE);
}

// ====== MQTT CALLBACK ======
void callback(char* topic, byte* payload, unsigned int length) {
  String msg = "";
  for (unsigned int i = 0; i < length; i++) {
    msg += (char)payload[i];
  }
  msg.trim();
  Serial.print("MQTT Received on [");
  Serial.print(topic);
  Serial.print("]: ");
  Serial.println(msg);

  // Alert logic: if payload exactly "s" -> activate alert
  if (msg.equalsIgnoreCase("s")) {
    alertActive = true;
    alertSource = msg;
    Serial.println("ALERT ACTIVATED");
  } else if (msg.equalsIgnoreCase("ok") || msg.equalsIgnoreCase("clear")) {
    // clear alert
    alertActive = false;
    alertSource = msg;
    digitalWrite(LED_PIN, LOW);
    ledState = LOW;
    Serial.println("ALERT CLEARED");
  } else {
    // Any other message: treat as clear (or optionally ignore)
    // Here I will clear alert for non-"s" messages, but you can change behavior if desired
    alertActive = false;
    alertSource = msg;
    digitalWrite(LED_PIN, LOW);
    ledState = LOW;
    Serial.println("ALERT CLEARED (other msg)");
  }
}

// ====== MQTT RECONNECT ======
void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting to MQTT... ");
    String dynamicID = clientID + String(random(0xffff), HEX);
    if (client.connect(dynamicID.c_str())) {
      Serial.println("Connected");
      client.subscribe(mqtt_topic);
    } else {
      Serial.print("failed rc=");
      Serial.println(client.state());
      delay(2000);
    }
  }
}

// ====== DRAW WARNING ICON ======
void drawWarningIcon(int x, int y, int size) {
  // Draw filled triangle (warning) and exclamation mark
  int half = size / 2;
  int h = (int)(size * 0.9);
  // triangle
  display.fillTriangle(x + size/2, y, x, y + h, x + size, y + h, SSD1306_WHITE);
  // exclamation (white background inside)
  display.fillRect(x + half - 2, y + 6, 4, h - 12, SSD1306_BLACK);
  display.fillRect(x + half - 1, y + h - 8, 2, 6, SSD1306_BLACK);
  // draw outline
  display.drawTriangle(x + size/2, y, x, y + h, x + size, y + h, SSD1306_WHITE);
}

// ====== DRAW ALERT SCREEN ======
void showAlertScreen() {
  display.clearDisplay();

  // Warning Icon (segitiga)
  display.fillTriangle(
    5, 5,     // kiri atas
    25, 5,    // kanan atas
    15, 20,   // bawah
    WHITE
  );

  // Teks Alert kecil
  display.setTextSize(2);
  display.setTextColor(WHITE);

  display.setCursor(20, 30);
  display.print("MOVEMENT");

  display.setCursor(20, 50);
  display.print("DETECTED!");


  display.display();
}

// ====== DRAW NORMAL SCREEN ======
void showNormalScreen(float tempVal, float humVal, int lightPerc, const String &timestamp) {
  display.clearDisplay();
  drawHeader("SIC 7|Team Incognito");

  // === Temperature (kiri) ===
  display.setTextSize(1);
  display.setCursor(5, 12);
  display.println("Temp");
  display.setTextSize(2);
  display.setCursor(5, 24);
  if (isnan(tempVal)) display.print("--");
  else display.print(tempVal, 1);
  display.setTextSize(1);
  display.print("C");

  // === Humidity (kanan) ===
  display.setTextSize(1);
  display.setCursor(70, 12);
  display.println("Hum");
  display.setTextSize(2);
  display.setCursor(70, 24);
  if (isnan(humVal)) display.print("--");
  else display.print(humVal, 0);
  display.setTextSize(1);
  display.print("%");

  // === Light (di bawah) ===
  display.setTextSize(1);
  display.setCursor(5, 50);
  display.print("Light: ");
  display.print(lightPerc);
  display.print("%");

  display.display();
}


void setup() {
  Serial.begin(115200);
  delay(50);
  dht.begin();

  // Initialize I2C (SDA=21, SCL=22)
  Wire.begin(21, 22);

  // OLED init
  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println("SSD1306 allocation failed");
    for (;;);
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // LED init
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  //Buzzer init
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  // Sensor pin
  pinMode(LDR_PIN, INPUT);

  // WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  int wifiRetries = 0;
  while (WiFi.status() != WL_CONNECTED && wifiRetries < 40) {
    delay(250);
    Serial.print(".");
    wifiRetries++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi connect failed (continuing, will retry in code) ");
  }

  // Setup NTP (GMT+7)
  configTime(gmtOffset_sec, daylightOffset_sec, "pool.ntp.org", "time.google.com");

  // MQTT client
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);

  // Initial splash
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 5);
  display.println("Mini IoT Project");
  display.println("SIC 7");
  display.println("Team Incognito");
  display.println("ESP32 Monitor");
  display.display();
  delay(1500);
}

void loop() {
  // Ensure WiFi & MQTT
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.reconnect();
  }
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // Read sensors
  float hum = dht.readHumidity();
  temperature = dht.readTemperature();
  ldrValue = analogRead(LDR_PIN);
  lightPercent = map(ldrValue, 0, 4095, 100, 0);

  String timestamp = getTimestamp();

  // Logging
Serial.print("Temp: ");
Serial.print(temperature);
Serial.print(" C | Hum: ");
Serial.print(hum);
Serial.print(" % | Light: ");
Serial.print(lightPercent);
Serial.println(" %");

  // Publish MQTT (with timestamp) every mqttPublishInterval
  if (millis() - lastMqttPublish >= mqttPublishInterval) {
      String payload = "{";
      payload += "\"timestamp\":\"" + timestamp + "\",";
      payload += "\"temperature\":" + String(temperature) + ",";
      payload += "\"humidity\":" + String(hum) + ",";
      payload += "\"light\":" + String(lightPercent);
      payload += "}";
    if (client.connected()) {
      client.publish(mqtt_topic, payload.c_str());
      Serial.print("Published: ");
      Serial.println(payload);
    }
    lastMqttPublish = millis();
  }

    if (alertActive) {
      unsigned long now = millis();

      // Blink LED & buzzer cepat
      if (now - lastLedToggle >= ledBlinkInterval) {
        lastLedToggle = now;

        ledState = !ledState;
        buzzerState = ledState;   // buzzer mengikuti LED blink

        digitalWrite(LED_PIN, ledState);
        digitalWrite(BUZZER_PIN, buzzerState);
      }

      showAlertScreen(); // tampilan alert
  }
  else {
      // pastikan LED & buzzer mati
      if (ledState || buzzerState) {
        ledState = LOW;
        buzzerState = LOW;
        digitalWrite(LED_PIN, LOW);
        digitalWrite(BUZZER_PIN, LOW);
      }
      
      // tampilan normal
      showNormalScreen(temperature, hum, lightPercent, timestamp);
  }

if(lightPercent <= 50){
  ledState = !ledState;
  digitalWrite(LED_PIN, ledState);
} else {
  ledState = LOW;
  digitalWrite(LED_PIN, LOW);
}

  // Short delay to avoid spamming; non-blocking style overall
  delay(200);
}
