from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Load the trained model, scaler, and encoders
MODEL_PATH = 'best_customer_classification_model.pkl'
SCALER_PATH = 'feature_scaler.pkl'
ENCODERS_PATH = 'label_encoders.pkl'
FEATURE_INFO_PATH = 'feature_info.pkl'

# Check if model files exist
if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH]):
    print("⚠️  WARNING: Model files not found! Please run train_optimized_model.py first.")
    model = None
    scaler = None
    FEATURE_COLS = None
else:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Try to load feature info
    if os.path.exists(FEATURE_INFO_PATH):
        feature_info = joblib.load(FEATURE_INFO_PATH)
        FEATURE_COLS = feature_info['feature_cols']
        TRAINING_THRESHOLDS = feature_info.get('training_thresholds', {})
        print("✓ Model loaded successfully!")
        print(f"✓ Using {len(FEATURE_COLS)} features")
    else:
        FEATURE_COLS = None
        print("✓ Model loaded but feature info not found")

# Options for categorical fields
CATEGORIES = {
    'Gender': ['Male', 'Female'],
    'Category': ['Clothing', 'Footwear', 'Accessories', 'Outerwear'],
    'Location': ['California', 'New York', 'Texas', 'Florida', 'Illinois', 'Nevada', 'Ohio', 
                 'Pennsylvania', 'Washington', 'Oregon', 'Arizona', 'Colorado', 'Georgia'],
    'Size': ['S', 'M', 'L', 'XL'],
    'Color': ['Red', 'Blue', 'Green', 'Black', 'White', 'Yellow', 'Pink', 'Purple', 'Gray', 'Orange'],
    'Season': ['Spring', 'Summer', 'Fall', 'Winter'],
    'Subscription Status': ['Yes', 'No'],
    'Shipping Type': ['Standard', 'Express', 'Free Shipping', 'Next Day Air', '2-Day Shipping', 'Store Pickup'],
    'Payment Method': ['Credit Card', 'PayPal', 'Debit Card', 'Venmo', 'Cash', 'Bank Transfer'],
    'Frequency of Purchases': ['Weekly', 'Fortnightly', 'Monthly', 'Quarterly', 'Annually', 'Bi-Weekly', 'Every 3 Months'],
    'Discount Applied': ['Yes', 'No'],
    'Promo Code Used': ['Yes', 'No']
}

@app.route('/')
def home():
    """Render the landing page"""
    return render_template('landing.html')

@app.route('/predict-form')
def predict_form():
    """Render the prediction form page"""
    # Always use simplified interface for the optimized model
    return render_template('index_simplified.html', categories=CATEGORIES)

@app.route('/test-clients')
def test_clients():
    """Render the test clients interface with auto-fill"""
    return render_template('index_with_presets.html', categories=CATEGORIES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first by running train_optimized_model.py'
            }), 500
        
        # Get form data
        data = request.json
        
        # Extract numerical features
        age_raw = float(data.get('age', 35))
        income_raw = float(data.get('income', 50000))
        previous_purchases_raw = int(data.get('previous_purchases', 10))
        review_rating_raw = float(data.get('review_rating', 3.5))
        num_web_visits_raw = int(data.get('num_web_visits', 5))
        
        # IMPORTANT: The CSV has pre-normalized numerical features
        # We need to normalize raw inputs to match the CSV's scale
        # Approximate normalization based on typical customer data ranges:
        # Age: mean~42, std~15
        # Income: mean~60000, std~30000  
        # Previous Purchases: mean~25, std~12
        # Review Rating: mean~3.75, std~0.7
        # NumWebVisitsMonth: mean~5, std~2.5
        
        age = (age_raw - 42) / 15
        income_normalized = (income_raw - 60000) / 30000
        previous_purchases = (previous_purchases_raw - 25) / 12
        review_rating = (review_rating_raw - 3.75) / 0.7
        num_web_visits = (num_web_visits_raw - 5) / 2.5
        
        # Map categorical inputs to encoded values (already numeric in CSV)
        # Gender: Male=1, Female=0
        gender = 1 if data.get('gender', 'Male') == 'Male' else 0
        
        # Category: Map to 0-3
        category_map = {'Clothing': 0, 'Footwear': 1, 'Accessories': 2, 'Outerwear': 3}
        category = category_map.get(data.get('category', 'Clothing'), 0)
        
        # Subscription Status: Yes=1, No=0
        subscription = 1 if data.get('subscription', 'No') == 'Yes' else 0
        
        # Discount Applied and Promo Code Used: Yes=1, No=0
        discount = 1 if data.get('discount', 'No') == 'Yes' else 0
        promo = 1 if data.get('promo', 'No') == 'Yes' else 0
        
        # Calculate engineered features using NORMALIZED values to match training
        # Income_Per_1k is just the normalized income (training uses df['Income'] directly)
        income_per_1k = income_normalized
        loyalty_score = previous_purchases * review_rating  # Product of normalized values
        engagement_level = num_web_visits * review_rating  # Product of normalized values
        price_sensitive = 1 if (discount == 1 or promo == 1) else 0
        
        # Create one-hot encoded features for Shipping Type
        shipping = data.get('shipping', 'Standard').lower()
        shipping_express = 1 if shipping == 'express' else 0
        shipping_free = 1 if shipping == 'free shipping' else 0
        shipping_nextday = 1 if shipping == 'next day air' else 0
        shipping_standard = 1 if shipping == 'standard' else 0
        shipping_pickup = 1 if shipping == 'store pickup' else 0
        
        # Create one-hot encoded features for Payment Method
        payment = data.get('payment', 'Credit Card').lower()
        payment_cash = 1 if payment == 'cash' else 0
        payment_credit = 1 if payment == 'credit card' else 0
        payment_debit = 1 if payment == 'debit card' else 0
        payment_paypal = 1 if payment == 'paypal' else 0
        payment_venmo = 1 if payment == 'venmo' else 0
        
        # Create one-hot encoded features for Frequency of Purchases
        frequency = data.get('frequency', 'Monthly').lower()
        freq_biweekly = 1 if frequency == 'bi-weekly' else 0
        freq_every3 = 1 if frequency == 'every 3 months' else 0
        freq_fortnightly = 1 if frequency == 'fortnightly' else 0
        freq_monthly = 1 if frequency == 'monthly' else 0
        freq_quarterly = 1 if frequency == 'quarterly' else 0
        freq_weekly = 1 if frequency == 'weekly' else 0
        
        # Create feature array with all 29 features in exact order
        features = np.array([[
            age,
            income_per_1k,
            previous_purchases,
            review_rating,
            num_web_visits,
            subscription,
            gender,
            category,
            discount,
            promo,
            loyalty_score,
            engagement_level,
            price_sensitive,
            shipping_express,
            shipping_free,
            shipping_nextday,
            shipping_standard,
            shipping_pickup,
            payment_cash,
            payment_credit,
            payment_debit,
            payment_paypal,
            payment_venmo,
            freq_biweekly,
            freq_every3,
            freq_fortnightly,
            freq_monthly,
            freq_quarterly,
            freq_weekly
        ]])
        
        # Note: The scaler was fit on already-normalized data from the CSV
        # So we need to apply proper normalization here to match training data format
        # OR we skip scaling since we'll normalize manually to match the CSV format
        
        # For now, use the scaler as-is (it will do minimal transformation)
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        # Determine segment
        if probability is not None:
            prob_low_value = float(probability[0])
            prob_high_value = float(probability[1])
            if prob_high_value > 0.7:
                segment = 'High Potential'
                color = 'success'
            elif prob_high_value > 0.3:
                segment = 'Medium Potential'
                color = 'warning'
            else:
                segment = 'Low Potential'
                color = 'danger'
        else:
            prob_high_value = float(prediction)
            prob_low_value = 1.0 - float(prediction)
            segment = 'High Value' if prediction == 1 else 'Not High Value'
            color = 'success' if prediction == 1 else 'danger'
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'probability': {
                'high_value': prob_high_value,
                'low_value': prob_low_value
            },
            'probability_display': f"{prob_high_value * 100:.1f}%",
            'segment': segment,
            'color': color,
            'recommendation': get_recommendation(segment, prob_high_value),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 400

def get_recommendation(segment, probability):
    """Generate marketing recommendations based on customer segment"""
    if segment == 'High Potential':
        return {
            'title': '🎯 High Priority Customer',
            'actions': [
                'Target with premium offers and exclusive deals',
                'Enroll in VIP loyalty program',
                'Personalized product recommendations',
                'Priority customer service'
            ]
        }
    elif segment == 'Medium Potential':
        return {
            'title': '📈 Medium Priority Customer',
            'actions': [
                'Send targeted promotional campaigns',
                'Offer limited-time discounts',
                'Encourage product reviews',
                'A/B test different marketing messages'
            ]
        }
    else:
        return {
            'title': '📧 Nurturing Customer',
            'actions': [
                'Low-cost email marketing campaigns',
                'Educational content and product guides',
                'Seasonal newsletters',
                'Re-engagement campaigns'
            ]
        }

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions from CSV upload"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # This endpoint can be extended for CSV file uploads
        return jsonify({'message': 'Batch prediction endpoint - Coming soon!'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'model_not_loaded',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 CUSTOMER VALUE PREDICTION API")
    print("="*80)
    print(f"Model Status: {'✓ Loaded' if model else '✗ Not Loaded'}")
    print("Server starting on http://127.0.0.1:5000")
    print("Press CTRL+C to quit")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
