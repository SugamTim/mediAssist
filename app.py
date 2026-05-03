from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import io
import base64
import traceback
import os

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

# ==================== GLOBAL VARIABLES ====================
# Symptom prediction variables
loaded_model = None
loaded_symptom_index = {}
loaded_label_encoder = None
symptom_list = []
df_description = pd.DataFrame()
df_causes_precautions = pd.DataFrame()
df_disease_severity = pd.DataFrame()

# Skin disease prediction variables
device = torch.device("cpu")
skin_model = None
skin_transform = None
skin_class_names = []
skin_disease_info = {}

# ==================== INITIALIZATION FUNCTIONS ====================
def initialize_symptom_predictor():
    """Initialize symptom prediction models and data"""
    global loaded_model, loaded_symptom_index, loaded_label_encoder
    global symptom_list, df_description, df_disease_severity, df_causes_precautions
    
    try:
        # Load trained model and encoders
        loaded_model = joblib.load("models/disease_predictor_model.joblib")
        loaded_symptom_index = joblib.load("utils/symptom_index.joblib")
        loaded_label_encoder = joblib.load("utils/label_encoder.joblib")
        print("✓ Symptom model and encoders loaded successfully")
    except Exception as e:
        print(f"✗ Error loading symptom model files: {e}")
        loaded_model = None
        loaded_symptom_index = {}
        loaded_label_encoder = None
    
    try:
        # Load datasets
        df_dataset = pd.read_csv('datasets/dataset.csv') 
        df_description = pd.read_csv('datasets/disease_description.csv')
        df_causes_precautions = pd.read_csv('datasets/disease_causes_precautions.csv')
        df_disease_severity = pd.read_csv('datasets/disease_severity.csv')
        print("✓ Symptom datasets loaded successfully")
    except Exception as e:
        print(f"✗ Error loading symptom CSV files: {e}")
        df_dataset = pd.DataFrame()
        df_description = pd.DataFrame()
        df_causes_precautions = pd.DataFrame()
        df_disease_severity = pd.DataFrame()
    
    # Prepare symptom list
    if loaded_symptom_index:
        symptom_list = list(loaded_symptom_index.keys())
        symptom_list = sorted(symptom_list)
        print(f"✓ Loaded {len(symptom_list)} symptoms from trained index")
    else:
        symptom_list = ['abdominal_pain', 'fatigue', 'nausea', 'loss_of_appetite', 'dark_urine', 'poor_appetite', 'yellowish_skin', 'pale_skin', 'seizures', 'yellowing_of_eyes', 'phlegm', 'blood_in_sputum', 'weight_loss', 'swelled_lymph_nodes', 'sweating', 'cough', 'mild_fever', 'night_sweats', 'breathlessness', 'diarrhoea', 'increased_sweating', 'anxiety', 'restlessness', 'mood_swings', 'excessive_hunger', 'fast_heart_rate', 'irregular_heartbeat', 'heat_intolerance', 'goiter', 'loss_of_balance', 'dizziness', 'poor_concentration', 'spinning_movements', 'chest_tightness', 'wheezing', 'sleep_disturbances', 'throat_irritation', 'mucoid_sputum', 'cramps', 'stomach_pain', 'vomiting', 'racing_heart', 'chest_pain', 'weakness_in_limbs', 'cold_sweats', 'hair_loss', 'neck_pain', 'headache', 'tingling_sensation', 'back_pain', 'joint_stiffness', 'reduced_joint_mobility', 'swelling_joints', 'hip_joint_pain', 'joint_pain', 'movement_stiffness', 'painful_walking', 'slurred_speech', 'weakness_of_one_body_side', 'altered_sensorium', 'pain_in_anal_region', 'pain_during_bowel_movements', 'bloating', 'irritation_in_anus', 'frequent_urination', 'painful_urination', 'continuous_feel_of_urine', 'burning_micturition', 'cloudy_urine', 'numbness', 'back_stiffness', 'reduced_grip_strength', 'shortness_of_breath', 'history_of_alcohol_consumption', 'swelling_of_stomach', 'distention_of_abdomen', 'bone_tenderness', 'severe_headache', 'muscle_pain', 'pain_behind_the_eyes', 'high_fever', 'nasal_congestion', 'runny_nose', 'congestion', 'hives', 'itching', 'watering_from_eyes', 'continuous_sneezing', 'sore_throat', 'knee_pain', 'limb_swelling', 'yellow_crust_ooze', 'red_sore_around_nose', 'blister', 'skin_rash', 'swollen_extremeties', 'muscle_weakness', 'dry_skin', 'cold_intolerance', 'depression', 'weight_gain', 'lethargy', 'cold_hands_and_feets', 'indigestion', 'bad_breath', 'blurred_and_distorted_vision', 'ringing_in_ears', 'sensitivity_to_light', 'malaise', 'receiving_unsterile_injections', 'blistering', 'red_spots_over_body', 'scurring', 'blackheads', 'pus_filled_pimples', 'skin_darkening', 'chills', 'bloody_stool', 'excessive_gas', 'constipation', 'excessive_thirst', 'slow_wound_healing', 'polyuria', 'acidity', 'hiccups', 'flushing', 'prominent_veins_on_calf', 'swollen_blood_vessels', 'swollen_ankles', 'swollen_legs', 'bruising', 'puffy_face_and_eyes', 'hoarseness', 'inflammatory_nails', 'small_dents_in_nails', 'skin_peeling', 'excessive_dryness', 'silver_like_dusting', 'loss_of_consciousness', 'nail_discoloration', 'morning_stiffness', 'enlarged_spleen', 'belly_pain', 'tremors', 'irritability', 'confusion', 'dehydration', 'receiving_blood_transfusion', 'night_chills', 'chills_with_rigor', 'spider_veins', 'lower_back_pain', 'ear_pain', 'nodal_skin_eruptions', 'dischromic_patches', 'peeling_skin', 'brain_fog', 'stiff_neck', 'eye_pain', 'rectal_pain', 'double_vision', 'palpitations', 'unsteadiness', 'bluish_lips', 'foul_smell_ofurine', 'bladder_discomfort', 'disorientation', 'general_weakness', 'rusty_sputum', 'yellow_urine', 'low_blood_pressure', 'increased_appetite', 'cold_extremities', 'patches_in_throat', 'frequent_infections', 'rectal_bleeding', 'dry_mouth', 'tarry_stools', 'family_history', 'muscle_wasting', 'visual_disturbances', 'eye_discharge', 'early_satiety', 'ulcers_on_tongue', 'skipped_heartbeat', 'shivering', 'muscle_spasms', 'agitation', 'shoulder_pain', 'hallucinations', 'pelvic_pain', 'blood_in_urine', 'toxic_look_(typhos)', 'fluid_overload', 'greasy_stools', 'brittle_nails', 'redness_of_eyes', 'bone_pain', 'dark_circles', 'spotting_urination', 'night_blindness', 'enlarged_lymph_nodes', 'irregular_sugar_level', 'acute_liver_failure', 'enlarged_thyroid', 'passage_of_gases', 'obesity', 'stretch_marks', 'urinary_incontinence', 'recurrent_fever', 'internal_itching', 'extra_marital_contacts', 'loss_of_bowel_control', 'sunken_eyes', 'excessive_drowsiness', 'panic_attacks', 'drying_and_tingling_lips', 'lack_of_concentration', 'coma', 'difficulty_swallowing', 'rectal_discharge', 'stomach_gurgling', 'nasal_polyps', 'fainting', 'loss_of_taste', 'sudden_weight_loss']
        
        def clean_symptom_name(symptom):
            if isinstance(symptom, str):
                return symptom.lower().strip()
            return ""
        
        symptom_list = [clean_symptom_name(s) for s in symptom_list if s]
        symptom_list = sorted(list(set(symptom_list)))
        print(f"✓ Loaded {len(symptom_list)} symptoms from datasets")

def initialize_skin_predictor():
    """Initialize skin disease prediction model and data"""
    global skin_model, skin_transform, skin_class_names, skin_disease_info
    
    # Skin disease class names
    skin_class_names = [
        'Acne', 'Actinic_Keratosis', 'Bullous', 'Candidiasis', 'DrugEruption',
        'Infestations_Bites', 'Lichen', 'Lupus', 'Moles', 'Psoriasis',
        'Rosacea', 'Seborrh_Keratoses', 'SkinCancer', 'Sun_Sunlight_Damage',
        'Tinea', 'Unknown', 'Vascular_Tumors', 'Vasculitis',
        'Vitiligo', 'Warts'
    ]
    
    # Load skin disease information
    try:
        skin_disease_df = pd.read_csv("datasets/skin_diseases.csv")
        skin_disease_info = {}
        for _, row in skin_disease_df.iterrows():
            skin_disease_info[row['Disease']] = {
                'description': row['Description'],
                'causes': row['Causes'],
                'precautions': [p.strip() for p in row['Precautions'].split(',')],
                'severity': row['Severity']
            }
        print("✓ Skin disease information loaded successfully")
    except Exception as e:
        print(f"✗ Error loading skin disease CSV: {e}")
        skin_disease_info = {}
    
    # Load skin disease model
    try:
        skin_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        NUM_CLASSES = len(skin_class_names)

        # Update the classifier head
        n_inputs = skin_model.classifier[2].in_features
        skin_model.classifier[2] = torch.nn.Linear(n_inputs, NUM_CLASSES)

        # Load the trained model
        state_dict = torch.load("models/skin_disease_classifier_model.pth", map_location=device)
        skin_model.load_state_dict(state_dict)
        skin_model.to(device)
        skin_model.eval()
        
        # Image preprocessing
        skin_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("✓ Skin disease model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading skin disease model: {e}")
        skin_model = None

# ==================== HELPER FUNCTIONS ====================
def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def prepare_input(selected_symptoms):
    """Convert selected symptoms to model input format using trained encoders"""
    if not loaded_symptom_index:
        raise ValueError("Trained symptom index not loaded")
    
    # Create zero vector with same length as training symptoms
    x = np.zeros(len(loaded_symptom_index))
    
    # Process each symptom
    for symptom in selected_symptoms:
        # Clean the symptom name (same as training)
        s = symptom.lower().strip()
        
        # Check if symptom exists in trained index
        if s in loaded_symptom_index:
            x[loaded_symptom_index[s]] = 1
    
    return x.reshape(1, -1)  # Reshape to (1, n_features)

def get_disease_description(disease_name):
    """Get description of the disease"""
    if df_description.empty:
        return f"Information about {disease_name}"
    
    disease_name_clean = str(disease_name).strip().lower()
    for idx, row in df_description.iterrows():
        if str(row['Disease']).strip().lower() == disease_name_clean:
            return str(row['Description'])
    
    return f"No detailed description available for {disease_name}"

def get_disease_precautions(disease_name):
    """Get precautions for the disease"""
    disease_name_clean = str(disease_name).strip().lower()
    for idx, row in df_causes_precautions.iterrows():
        if str(row['Disease']).strip().lower() == disease_name_clean:
            precautions = str(row['Precautions'])
            return precautions

def get_disease_causes(disease_name):
    """Get causes for the disease"""
    disease_name_clean = str(disease_name).strip().lower()
    for idx, row in df_causes_precautions.iterrows():
        if str(row['Disease']).strip().lower() == disease_name_clean:
            causes = str(row['Causes'])
            return causes

def get_disease_severity(disease_name):
    """Show disease severity"""
    disease_name = disease_name.lower().strip()
    
    match = df_disease_severity[df_disease_severity["Disease"] == disease_name]
    
    if not match.empty:
        return match.iloc[0]["Severity"]
    
    return "Medium"  # default fallback

def get_doctor_recommendation(disease_name):
    """Get recommended doctor type based on disease"""
    doctor_types = {
        'dermatologist': ['Acne', 'Psoriasis', 'Fungal infection', 'Impetigo', 'Chicken pox'],
        'endocrinologist': ['Hyperthyroidism', 'Hypothyroidism', 'Diabetes', 'Hypoglycemia'],
        'infectious disease specialist': ['AIDS', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C', 
                                          'Hepatitis D', 'Hepatitis E', 'Tuberculosis', 'Typhoid', 
                                          'Dengue', 'Malaria', 'Pneumonia'],
        'gastroenterologist': ['Chronic cholestasis', 'Jaundice', 'Peptic Ulcer Diseae', 
                               'GERD', 'Alcoholic hepatitis', 'Gastroenteritis'],
        'cardiologist': ['Hypertension', 'Heart attack'],
        'neurologist': ['Migraine', '(vertigo) Paroymsal Positional Vertigo', 
                        'Paralysis (brain hemorrhage)'],
        'rheumatologist': ['Arthritis', 'Osteoarthristis', 'Cervical spondylosis', 
                           'Varicose veins', 'Bronchial Asthma'],
        'urologist': ['Urinary tract infection'],
        'allergist': ['Allergy', 'Common Cold'],
        'proctologist': ['Dimorphic Hemmorhoids(Piles)'],
        'general physician': ['Drug Reaction']
    }
    
    primary_disease = str(disease_name).lower()
    recommended_doctor = 'general physician'
    
    for doctor, keywords in doctor_types.items():
        if any(keyword.lower() in primary_disease for keyword in keywords):
            recommended_doctor = doctor
            break
    
    return recommended_doctor.title()

# ==================== ROUTES ====================
@app.route('/')
def home():
    """Render the main page with both functionalities"""
    return render_template('index.html')

@app.route('/predict_symptom', methods=['POST'])
def predict_symptom():
    """Handle disease prediction using symptoms"""
    try:
        if not loaded_model:
            return jsonify({
                'success': False,
                'error': 'Prediction model not loaded. Please check server configuration.'
            })
        
        # Get symptoms from form
        selected_symptoms = request.form.getlist('symptoms[]')
        
        if not selected_symptoms:
            return jsonify({
                'success': False,
                'error': 'Please select at least one symptom'
            })
        
        print(f"Selected symptoms: {selected_symptoms}")
        
        # Prepare input using trained encoders
        input_data = prepare_input(selected_symptoms)
        
        # Make prediction using trained model
        probabilities = loaded_model.predict_proba(input_data)[0]
        
        # Get top 3 predictions with confidence
        top_n = 3
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_probs = probabilities[top_indices]
        
        # Normalize probabilities to sum to 100%
        normalized_probs = top_probs / top_probs.sum()
        
        # Prepare predictions with confidence scores
        predictions = []
        for idx, prob in zip(top_indices, normalized_probs):
            disease_name = loaded_label_encoder.inverse_transform([idx])[0]
            confidence = round(prob * 100, 2)
            
            predictions.append({
                'disease': str(disease_name).title(),
                'confidence': float(confidence),
                'description': get_disease_description(disease_name),
                'causes': get_disease_causes(disease_name),
                'precautions': get_disease_precautions(disease_name),
                'severity': get_disease_severity(disease_name)
            })
        
        # Get recommended doctor
        recommended_doctor = get_doctor_recommendation(predictions[0]['disease'])
        
        # Prepare response
        response_data = {
            'success': True,
            'predictions': convert_numpy_types(predictions),
            'selected_symptoms': selected_symptoms,
            'recommended_doctor': recommended_doctor,
            'total_confidence': float(round(sum([p['confidence'] for p in predictions]), 2))
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in predict_symptom(): {e}")
        print(f"Error details:\n{traceback.format_exc()}")
        
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        })

@app.route('/predict_skin', methods=['POST'])
def predict_skin():
    """Handle skin disease prediction from image"""
    if not skin_model:
        return jsonify({
            'success': False,
            'error': 'Skin disease model not loaded'
        })
    
    if "image" not in request.files:
        return jsonify({
            'success': False,
            "error": "No image uploaded"
        })

    file = request.files["image"]
    if file.filename == "":
        return jsonify({
            'success': False,
            "error": "No image selected"
        })

    try:
        # Read image in memory
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess and predict
        image_tensor = skin_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = skin_model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        prediction = skin_class_names[pred.item()]
        confidence_value = confidence.item() * 100

        # Get disease information
        info = skin_disease_info.get(prediction, {
            'description': 'No description available.',
            'causes': 'Causes not specified.',
            'precautions': ['Consult a dermatologist for proper diagnosis and treatment.'],
            'severity': 'Unknown'
        })

        # Convert image to Base64 for preview (JPEG, smaller memory)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_data = f"data:image/jpeg;base64,{img_str}"

        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': f"{confidence_value:.2f}%",
            'description': info['description'],
            'causes': info['causes'],
            'precautions': info['precautions'],
            'severity': info['severity'],
            'img_data': img_data
        })

    except Exception as e:
        print(f"Error during skin prediction: {e}")
        return jsonify({
            'success': False,
            "error": str(e)
        })

# ==================== SYMPTOM MANAGEMENT ROUTES ====================
@app.route('/get_all_symptoms', methods=['GET'])
def get_all_symptoms():
    """Get all available symptoms"""
    return jsonify({
        'symptoms': symptom_list,
        'count': len(symptom_list)
    })

@app.route('/search_symptoms', methods=['POST'])
def search_symptoms():
    """Search for symptoms based on query"""
    try:
        data = request.get_json()
        query = data.get('query', '').lower() if data else ''
        
        if not query:
            return jsonify({'symptoms': []})
        
        matched_symptoms = [s for s in symptom_list if query in s.lower()]
        
        return jsonify({
            'symptoms': matched_symptoms[:10]
        })
    except Exception as e:
        return jsonify({'symptoms': [], 'error': str(e)})

# ==================== SKIN DISEASE ROUTES ====================
@app.route('/get_skin_classes', methods=['GET'])
def get_skin_classes():
    """Get skin disease classes (API endpoint)"""
    return jsonify({
        'classes': skin_class_names,
        'count': len(skin_class_names)
    })

# ==================== HEALTH CHECK ====================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    symptom_model_loaded = loaded_model is not None
    skin_model_loaded = skin_model is not None
    
    return jsonify({
        'status': 'healthy',
        'symptom_model': 'loaded' if symptom_model_loaded else 'not loaded',
        'skin_model': 'loaded' if skin_model_loaded else 'not loaded',
        'symptom_count': len(symptom_list),
        'skin_class_count': len(skin_class_names)
    })

# ==================== MAIN ENTRY POINT ====================
if __name__ == '__main__':
    print("=" * 50)
    print("MediAssist - Medical Diagnosis AI System")
    print("=" * 50)
    
    # Initialize both systems
    initialize_symptom_predictor()
    initialize_skin_predictor()
    
    print("\nSystem Status:")
    print(f"  Symptom Model: {'✓ Loaded' if loaded_model is not None else '✗ Not loaded'}")
    print(f"  Skin Model: {'✓ Loaded' if skin_model is not None else '✗ Not loaded'}")
    print(f"  Symptoms Available: {len(symptom_list)}")
    print(f"  Skin Classes: {len(skin_class_names)}")
    print("=" * 50)
    print("Server starting on http://localhost:5000")
    print("=" * 50)
    
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
