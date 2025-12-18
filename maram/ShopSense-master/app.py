from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# Add recommand_model to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'recommand_model'))

app = Flask(__name__)

# Load the trained model, scaler, and encoders
MODEL_PATH = 'best_customer_classification_model.pkl'
SCALER_PATH = 'feature_scaler.pkl'
ENCODERS_PATH = 'label_encoders.pkl'
FEATURE_INFO_PATH = 'feature_info.pkl'

# Check if model files exist
if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODERS_PATH]):
    print("⚠️  WARNING: Model files not found! Please run train_simplified_model.py first.")
    model = None
    scaler = None
    label_encoders = None
    FEATURE_COLS = None
else:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    
    # Try to load feature info (for simplified model)
    if os.path.exists(FEATURE_INFO_PATH):
        feature_info = joblib.load(FEATURE_INFO_PATH)
        FEATURE_COLS = feature_info['feature_cols']
        TRAINING_THRESHOLDS = feature_info.get('training_thresholds', {})
        print("✓ Simplified model loaded successfully!")
        print(f"✓ Using {len(FEATURE_COLS)} features: {FEATURE_COLS}")
    else:
        # Fallback to old feature set if feature_info not found
        FEATURE_COLS = [
            'Age', 'Income', 'NumWebVisitsMonth', 'Review Rating', 'Previous Purchases',
            'Gender_encoded', 'Category_encoded', 'Location_encoded', 'Size_encoded',
            'Color_encoded', 'Season_encoded', 'Subscription Status_encoded',
            'Shipping Type_encoded', 'Payment Method_encoded', 'Frequency of Purchases_encoded',
            'Discount Applied_encoded', 'Promo Code Used_encoded',
            'Income_to_Purchase_Ratio', 'Web_Engagement', 'Loyalty_Score', 'Age_Group_encoded'
        ]
        print("✓ Model loaded successfully!")

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
    # Check if using simplified model
    if FEATURE_COLS and (len(FEATURE_COLS) <= 10 and 'Loyalty_Score' in FEATURE_COLS):
        return render_template('index_simplified.html', categories=CATEGORIES)
    else:
        return render_template('index.html', categories=CATEGORIES)

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
                'error': 'Model not loaded. Please train the model first by running train_simplified_model.py'
            }), 500
        
        # Get form data
        data = request.json
        
        # Check if using optimized 14-feature model
        if len(FEATURE_COLS) == 14:
            # OPTIMIZED MODEL - 14 essential features
            income = float(data.get('income', 50000))
            age = float(data.get('age', 35))
            previous_purchases = int(data.get('previous_purchases', 10))
            review_rating = float(data.get('review_rating', 3.5))
            num_web_visits = int(data.get('num_web_visits', 5))
            
            # Encode categorical features
            gender_encoded = label_encoders['Gender'].transform([data.get('gender', 'Male')])[0]
            category_encoded = label_encoders['Category'].transform([data.get('category', 'Clothing')])[0]
            subscription_encoded = label_encoders['Subscription Status'].transform([data.get('subscription', 'No')])[0]
            shipping_encoded = label_encoders['Shipping Type'].transform([data.get('shipping', 'Standard')])[0]
            discount_encoded = label_encoders['Discount Applied'].transform([data.get('discount', 'No')])[0]
            promo_encoded = label_encoders['Promo Code Used'].transform([data.get('promo', 'No')])[0]
            
            # Calculate engineered features (must match training exactly)
            income_per_1k = income / 1000
            loyalty_score = previous_purchases * review_rating
            engagement_level = num_web_visits * review_rating
            price_sensitive = 1 if (data.get('discount', 'No') == 'Yes' or data.get('promo', 'No') == 'Yes') else 0
            
            # Create feature array with 14 features in exact order
            features = np.array([[
                age,
                income_per_1k,
                previous_purchases,
                review_rating,
                num_web_visits,
                subscription_encoded,
                gender_encoded,
                category_encoded,
                shipping_encoded,
                discount_encoded,
                promo_encoded,
                loyalty_score,
                engagement_level,
                price_sensitive
            ]])
        # Check if using enhanced model with 23 features
        elif len(FEATURE_COLS) >= 23:
            # ENHANCED MODEL - Full feature set with composite scoring
            income = float(data.get('income', 50000))
            age = float(data.get('age', 35))
            previous_purchases = int(data.get('previous_purchases', 10))
            review_rating = float(data.get('review_rating', 3.5))
            num_web_visits = int(data.get('num_web_visits', 5))
            
            # Encode categorical features
            gender_encoded = label_encoders['Gender'].transform([data.get('gender', 'Male')])[0]
            category_encoded = label_encoders['Category'].transform([data.get('category', 'Clothing')])[0]
            subscription_encoded = label_encoders['Subscription Status'].transform([data.get('subscription', 'No')])[0]
            shipping_encoded = label_encoders['Shipping Type'].transform([data.get('shipping', 'Standard')])[0]
            payment_encoded = label_encoders['Payment Method'].transform([data.get('payment', 'Credit Card')])[0]
            
            discount_encoded = 1 if data.get('discount', 'No') == 'Yes' else 0
            promo_encoded = 1 if data.get('promo', 'No') == 'Yes' else 0
            
            # Calculate engineered features (must match training exactly)
            income_per_1k = income / 1000
            
            # Age group
            if age <= 30:
                age_group = 0
            elif age <= 45:
                age_group = 1
            elif age <= 60:
                age_group = 2
            else:
                age_group = 3
            
            loyalty_score = previous_purchases * review_rating
            age_income_interaction = (age / 100) * (income / 10000)
            engagement_score = num_web_visits * review_rating
            purchase_frequency_score = previous_purchases / (age - 17)  # Purchases per year since 18
            
            # Boolean features (using training statistics for thresholds)
            high_engagement = 1 if num_web_visits > TRAINING_THRESHOLDS.get('web_visits_median', 5) else 0
            frequent_buyer = 1 if previous_purchases > TRAINING_THRESHOLDS.get('purchases_median', 20) else 0
            premium_customer = 1 if (subscription_encoded == 1 and previous_purchases > TRAINING_THRESHOLDS.get('purchases_q25', 15)) else 0
            discount_dependent = 1 if (discount_encoded == 1 and promo_encoded == 1) else 0
            high_income = 1 if income > TRAINING_THRESHOLDS.get('income_median', 50000) else 0
            satisfied_customer = 1 if review_rating >= 4.0 else 0
            
            # Create feature array with 23 features in exact order
            features = np.array([[
                income_per_1k,
                age,
                age_group,
                previous_purchases,
                review_rating,
                num_web_visits,
                gender_encoded,
                category_encoded,
                subscription_encoded,
                shipping_encoded,
                payment_encoded,
                discount_encoded,
                promo_encoded,
                loyalty_score,
                age_income_interaction,
                engagement_score,
                purchase_frequency_score,
                high_engagement,
                frequent_buyer,
                premium_customer,
                discount_dependent,
                high_income,
                satisfied_customer
            ]])
        # Check if using enhanced model with many features
        elif len(FEATURE_COLS) >= 17:
            # ENHANCED MODEL - Full feature set
            income = float(data.get('income', 50000))
            age = float(data.get('age', 35))
            previous_purchases = int(data.get('previous_purchases', 10))
            review_rating = float(data.get('review_rating', 3.5))
            num_web_visits = int(data.get('num_web_visits', 5))
            
            # Encode categorical features
            gender_encoded = label_encoders['Gender'].transform([data.get('gender', 'Male')])[0]
            category_encoded = label_encoders['Category'].transform([data.get('category', 'Clothing')])[0]
            subscription_encoded = label_encoders['Subscription Status'].transform([data.get('subscription', 'No')])[0]
            shipping_encoded = label_encoders['Shipping Type'].transform([data.get('shipping', 'Standard')])[0]
            payment_encoded = label_encoders['Payment Method'].transform([data.get('payment', 'Credit Card')])[0]
            
            discount_encoded = 1 if data.get('discount', 'No') == 'Yes' else 0
            promo_encoded = 1 if data.get('promo', 'No') == 'Yes' else 0
            
            # Calculate engineered features
            income_per_1k = income / 1000
            if age <= 30:
                age_group = 0
            elif age <= 45:
                age_group = 1
            elif age <= 60:
                age_group = 2
            else:
                age_group = 3
            
            loyalty_score = previous_purchases * review_rating
            high_engagement = 1 if num_web_visits > 5 else 0
            premium_customer = 1 if (subscription_encoded == 1 and previous_purchases > 10) else 0
            value_seeker = 1 if (discount_encoded == 1 or promo_encoded == 1) else 0
            engagement_score = num_web_visits * review_rating
            
            # Create feature array
            features = np.array([[
                income_per_1k,
                age,
                age_group,
                previous_purchases,
                review_rating,
                num_web_visits,
                gender_encoded,
                category_encoded,
                subscription_encoded,
                shipping_encoded,
                payment_encoded,
                discount_encoded,
                promo_encoded,
                loyalty_score,
                high_engagement,
                premium_customer,
                value_seeker,
                engagement_score
            ]])
        # Check if using simplified model (8 features - NO Purchase Amount)
        elif len(FEATURE_COLS) == 8 and 'Loyalty_Score' in FEATURE_COLS:
            # SIMPLIFIED MODEL - 8 predictive features
            income = float(data.get('income', 50000))
            previous_purchases = int(data.get('previous_purchases', 10))
            review_rating = float(data.get('review_rating', 3.5))
            age = float(data.get('age', 35))
            num_web_visits = int(data.get('num_web_visits', 5))
            subscription_encoded = label_encoders['Subscription Status'].transform([data.get('subscription', 'No')])[0]
            
            # Calculate engineered features
            loyalty_score = previous_purchases * review_rating
            income_purchasing_power = income / 10000
            
            # Create feature array with 8 features
            features = np.array([[
                income,
                previous_purchases,
                review_rating,
                age,
                num_web_visits,
                subscription_encoded,
                loyalty_score,
                income_purchasing_power
            ]])
        # Check if using old simplified model (4 features)
        elif FEATURE_COLS == ['Income', 'Previous Purchases', 'Review Rating', 'Subscription_encoded']:
            # OLD SIMPLIFIED MODEL - Only 4 predictive features
            income = float(data.get('income', 50000))
            previous_purchases = int(data.get('previous_purchases', 10))
            review_rating = float(data.get('review_rating', 3.5))
            subscription_encoded = label_encoders['Subscription Status'].transform([data.get('subscription', 'No')])[0]
            
            # Create feature array with only 4 features
            features = np.array([[
                income,
                previous_purchases,
                review_rating,
                subscription_encoded
            ]])
        else:
            # ORIGINAL MODEL - 21 features
            # Extract and validate numerical features
            age = float(data.get('age', 30))
            income = float(data.get('income', 50000))
            num_web_visits = int(data.get('num_web_visits', 5))
            review_rating = float(data.get('review_rating', 3.5))
            previous_purchases = int(data.get('previous_purchases', 10))
            purchase_amount = float(data.get('purchase_amount', 50))
            
            # Encode categorical features
            gender_encoded = label_encoders['Gender'].transform([data.get('gender', 'Male')])[0]
            category_encoded = label_encoders['Category'].transform([data.get('category', 'Clothing')])[0]
            location_encoded = label_encoders['Location'].transform([data.get('location', 'California')])[0]
            size_encoded = label_encoders['Size'].transform([data.get('size', 'M')])[0]
            color_encoded = label_encoders['Color'].transform([data.get('color', 'Blue')])[0]
            season_encoded = label_encoders['Season'].transform([data.get('season', 'Spring')])[0]
            subscription_encoded = label_encoders['Subscription Status'].transform([data.get('subscription', 'No')])[0]
            shipping_encoded = label_encoders['Shipping Type'].transform([data.get('shipping', 'Standard')])[0]
            payment_encoded = label_encoders['Payment Method'].transform([data.get('payment', 'Credit Card')])[0]
            frequency_encoded = label_encoders['Frequency of Purchases'].transform([data.get('frequency', 'Monthly')])[0]
            
            # Encode binary features
            discount_encoded = 1 if data.get('discount', 'No') == 'Yes' else 0
            promo_encoded = 1 if data.get('promo', 'No') == 'Yes' else 0
            
            # Calculate engineered features
            income_to_purchase_ratio = income / (purchase_amount + 1)
            web_engagement = num_web_visits * review_rating
            loyalty_score = previous_purchases * review_rating
            
            # Age group encoding
            if age <= 25:
                age_group_encoded = 0  # Young
            elif age <= 40:
                age_group_encoded = 1  # Adult
            elif age <= 60:
                age_group_encoded = 2  # Middle
            else:
                age_group_encoded = 3  # Senior
            
            # Create feature array in correct order
            features = np.array([[
                age, income, num_web_visits, review_rating, previous_purchases,
                gender_encoded, category_encoded, location_encoded, size_encoded,
                color_encoded, season_encoded, subscription_encoded,
                shipping_encoded, payment_encoded, frequency_encoded,
                discount_encoded, promo_encoded,
                income_to_purchase_ratio, web_engagement, loyalty_score, age_group_encoded
            ]])
        
        # Scale features
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
        
        # Prepare response with correct format for both interfaces
        result = {
            'prediction': int(prediction),  # 0 or 1
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
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 400

@app.route('/segment', methods=['POST'])
def segment():
    """Handle customer segmentation/clustering requests"""
    try:
        data = request.json
        print(f"DEBUG: Segmentation request received: {data}")
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract features in the order expected by the clustering model
        features = ['Income', 'Age', 'Wines', 'Fruits', 'Meat', 'Fish', 'Sweet', 'Gold']
        
        try:
            feature_values = [float(data.get(f, 0)) for f in features]
            print(f"DEBUG: Feature values (raw): {feature_values}")
        except (ValueError, TypeError) as e:
            print(f"DEBUG: ValueError in feature parsing: {e}")
            return jsonify({"error": "Invalid feature values"}), 400
        
        # Determine expected number of features
        expected_n_features = None
        if clustering_scaler is not None and hasattr(clustering_scaler, 'n_features_in_'):
            expected_n_features = int(clustering_scaler.n_features_in_)
        
        if expected_n_features is None:
            expected_n_features = 23  # Fallback
        
        # Pad with zeros if needed
        if len(feature_values) < expected_n_features:
            pad_len = expected_n_features - len(feature_values)
            feature_values = feature_values + [0.0] * pad_len
            print(f"DEBUG: Padded feature_values to {expected_n_features} items")
        elif len(feature_values) > expected_n_features:
            feature_values = feature_values[:expected_n_features]
            print(f"DEBUG: Truncated feature_values to {expected_n_features} items")
        
        # If models are not loaded, return fallback
        if not gmm_model or not clustering_scaler:
            print("DEBUG: Clustering models not loaded — returning fallback")
            return jsonify({
                "cluster": 0,
                "meaning": "Model not available",
                "probability": 0.0
            })
        
        # Scale and predict cluster
        data_scaled = clustering_scaler.transform([feature_values])
        cluster = int(gmm_model.predict(data_scaled)[0])
        probability = float(np.max(gmm_model.predict_proba(data_scaled)))
        meaning = cluster_meanings.get(cluster, "Unknown profile")
        
        response_data = {
            "cluster": cluster,
            "meaning": meaning,
            "probability": probability
        }
        print(f"DEBUG: Segmentation response: {response_data}")
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"DEBUG: Exception in segment: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

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
        'recommendation_loaded': recommendation_system_loaded,
        'timestamp': datetime.now().isoformat()
    })

# ============================================================================
# RECOMMENDATION SYSTEM ROUTES
# ============================================================================

# Load recommendation system
recommendation_system_loaded = False
try:
    from predict import get_recommendations, get_available_items, get_customer_info, get_segment_popular_items
    recommendation_system_loaded = True
    print("✓ Recommendation system loaded successfully!")
except Exception as e:
    print(f"⚠️  WARNING: Recommendation system not loaded: {e}")
    print("   Run: python recommand_model/train_model.py to train the model")

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get product recommendations for a customer"""
    try:
        if not recommendation_system_loaded:
            return jsonify({
                'error': 'Recommendation system not loaded. Please train the model first.',
                'status': 'error'
            }), 500
        
        data = request.json
        customer_id = int(data.get('customer_id', 1))
        top_n = int(data.get('top_n', 5))
        
        # Get recommendations
        result = get_recommendations(customer_id=customer_id, top_n=top_n)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Recommendation failed: {str(e)}',
            'status': 'error'
        }), 400

@app.route('/segment-info', methods=['POST'])
def segment_info():
    """Get popular items in a customer segment"""
    try:
        if not recommendation_system_loaded:
            return jsonify({
                'error': 'Recommendation system not loaded',
                'status': 'error'
            }), 500
        
        data = request.json
        segment_id = int(data.get('segment_id', 0))
        top_n = int(data.get('top_n', 10))
        
        result = get_segment_popular_items(segment_id=segment_id, top_n=top_n)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/customer-info/<int:customer_id>')
def customer_info(customer_id):
    """Get information about a specific customer"""
    try:
        if not recommendation_system_loaded:
            return jsonify({
                'error': 'Recommendation system not loaded',
                'status': 'error'
            }), 500
        
        result = get_customer_info(customer_id)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/available-items')
def available_items():
    """Get list of all available items"""
    try:
        if not recommendation_system_loaded:
            return jsonify({
                'error': 'Recommendation system not loaded',
                'status': 'error'
            }), 500
        
        items = get_available_items()
        return jsonify({
            'items': items,
            'count': len(items),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 CUSTOMER VALUE PREDICTION API")
    print("="*80)
    print(f"Model Status: {'✓ Loaded' if model else '✗ Not Loaded'}")
    print(f"Recommendation System: {'✓ Loaded' if recommendation_system_loaded else '✗ Not Loaded'}")
    print("Server starting on http://127.0.0.1:5000")
    print("Press CTRL+C to quit")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
