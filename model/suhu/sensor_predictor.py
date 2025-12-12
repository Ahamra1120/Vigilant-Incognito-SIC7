
class SensorMLPredictor:
    def __init__(self, model_path=None):
        """
        Initialize ML predictor for temperature and humidity
        """
        self.model_path = model_path or "sensor_model.pkl"
        self.scaler_path = "scaler.pkl"
        self.label_encoder_path = "label_encoder.pkl"
        
        # Initialize models
        self.classifier = None
        self.anomaly_detector = None
        self.scaler = None
        self.label_encoder = None
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            self.classifier = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            with open(self.label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("✅ ML models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def predict(self, temperature, humidity):
        """Make prediction for given temperature and humidity"""
        # Prepare features
        temp_humidity_ratio = temperature / (humidity + 0.1)
        comfort_index = 0.5 * temperature + 0.5 * humidity
        
        features = np.array([[temperature, humidity, temp_humidity_ratio, comfort_index]])
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
        
        # Anomaly detection
        anomaly_score = self.anomaly_detector.score_samples(features_scaled)[0]
        is_anomaly = anomaly_score < -0.2
        
        return {
            'condition': prediction,
            'confidence': float(confidence),
            'anomaly_score': float(anomaly_score),
            'is_anomaly': is_anomaly,
            'probabilities': {label: float(prob) for label, prob in zip(self.classifier.classes_, probabilities)}
        }
