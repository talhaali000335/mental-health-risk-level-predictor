# from flask import Flask, render_template, request, jsonify
# import joblib
# import pandas as pd
# import numpy as np

# app = Flask(__name__)

# # load Models and Scalers 
# try:
#     maternal_model = joblib.load('maternal_health_gbc_model.pkl')
#     maternal_scaler = joblib.load('scaler.pkl')
#     fetal_model = joblib.load('fetal_health_model.pkl')
#     fetal_scaler = joblib.load('fetal_scaler.pkl')
# except Exception as e:
#     print(f"Error loading models: {e}. Make sure .pkl files are in the same folder.")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict_maternal', methods=['POST'])
# def predict_maternal():
#     try:
#         data = request.form
#         age = float(data['age'])
#         diastolic_bp = float(data['diastolic_bp'])
#         bs = float(data['bs'])
#         body_temp = float(data['body_temp'])
#         heart_rate = float(data['heart_rate'])
        
#         maternal_columns = ["Age", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
#         custom_data = pd.DataFrame([[age, diastolic_bp, bs, body_temp, heart_rate]], 
#                                    columns=maternal_columns)
        
#         # Data scale 
#         scaled_data_array = maternal_scaler.transform(custom_data)
        
#         scaled_data_df = pd.DataFrame(scaled_data_array, columns=maternal_columns)
        
#         prediction = maternal_model.predict(scaled_data_df)[0]
        
#         risk_labels = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
#         result = risk_labels[prediction]
        
#         action_statements = {
#             "Low Risk": "Maternal parameters are within normal clinical thresholds. Keep up with routine checkups and balanced hydration.",
#             "Mid Risk": "Mild deviations detected in blood sugar or pressure metrics. Schedule a routine gynecologist consultation in 24-48 hours.",
#             "High Risk": "Critical maternal vital signs identified. Immediate medical intervention required! Contact emergency services or your doctor instantly."
#         }
        
#         return jsonify({
#             'status': 'success', 
#             'result': result,
#             'action': action_statements[result]
#         })
        
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)})

# @app.route('/predict_fetal', methods=['POST'])
# def predict_fetal():
#     try:
#         data = request.form
        
#         # User Inputs (Strictly float)
#         baseline = float(data['baseline_value'])
#         accelerations = float(data['accelerations'])
#         movement = float(data['fetal_movement'])
#         contractions = float(data['uterine_contractions'])
        
#         # Exact 21 Columns 
#         fetal_columns = [
#             'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 
#             'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 
#             'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
#             'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability', 
#             'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
#             'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median', 
#             'histogram_variance', 'histogram_tendency'
#         ]
        
#         # Medical base-line values default dictionary
#         input_dict = {col: 0.0 for col in fetal_columns}
        
#         # Histogram and general metrics 
#         input_dict['abnormal_short_term_variability'] = 35.0   # Normal text range
#         input_dict['mean_value_of_short_term_variability'] = 1.3
#         input_dict['histogram_mode'] = baseline
#         input_dict['histogram_mean'] = baseline
#         input_dict['histogram_median'] = baseline
        
#         # User inputs map 
#         input_dict['baseline value'] = baseline
#         input_dict['accelerations'] = accelerations
#         input_dict['fetal_movement'] = movement
#         input_dict['uterine_contractions'] = contractions
        
#         # made DataFrame
#         fetal_data = pd.DataFrame([input_dict], columns=fetal_columns)
        
#         # Scaler transform 
#         scaled_fetal = fetal_scaler.transform(fetal_data)
        
#         scaled_df = pd.DataFrame(scaled_fetal, columns=fetal_columns)
#         prediction = fetal_model.predict(scaled_df)[0]
        
#         # Mapping Results (1: Normal, 2: Suspect, 3: Pathological)
#         fetal_labels = {1: "Normal", 2: "Suspect", 3: "Pathological"}
#         result = fetal_labels.get(prediction, str(prediction))
        
#         # Clinical Action Statements For Fetal Health
#         action_statements = {
#             "Normal": "Excellent baseline heart rate and stable fetal activity recorded. Continue monitoring routine kick-counts.",
#             "Suspect": "Atypical cardiotocography (CTG) patterns detected. Rest well, track movements, and repeat the CTG test soon.",
#             "Pathological": "Severe fetal distress or oxygen saturation warning. Critical Emergency! Report to the labor room immediately."
#         }
        
#         return jsonify({
#             'status': 'success', 
#             'result': result,
#             'action': action_statements.get(result, "Check configuration parameters.")
#         })
        
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)


from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import joblib
from pipeline import ClinicalInferencePipeline

app = FastAPI(title="Women Health Care AI Platform")
templates = Jinja2Templates(directory="templates")

# Safe asset initialization
maternal_model = joblib.load('maternal_health_gbc_model.pkl')
maternal_scaler = joblib.load('scaler.pkl')
fetal_model = joblib.load('fetal_health_model.pkl')
fetal_scaler = joblib.load('fetal_scaler.pkl')

# pipeline initialization
medical_pipeline = ClinicalInferencePipeline(maternal_model, maternal_scaler, fetal_model, fetal_scaler)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict_maternal")
async def predict_maternal(
    age: float = Form(...),
    diastolic_bp: float = Form(...),
    bs: float = Form(...),
    body_temp: float = Form(...),
    heart_rate: float = Form(...)
):
    try:
        result = medical_pipeline.process_maternal(age, diastolic_bp, bs, body_temp, heart_rate)
        
        action_statements = {
            "Low Risk": "Maternal parameters are within normal clinical thresholds. Keep up with routine checkups and balanced hydration.",
            "Mid Risk": "Mild deviations detected in blood sugar or pressure metrics. Schedule a routine gynecologist consultation in 24-48 hours.",
            "High Risk": "Critical maternal vital signs identified. Immediate medical intervention required! Contact emergency services or your doctor instantly."
        }
        
        return JSONResponse({
            'status': 'success', 
            'result': result,
            'action': action_statements[result]
        })
    except Exception as e:
        return JSONResponse({'status': 'error', 'message': str(e)})

@app.post("/predict_fetal")
async def predict_fetal(
    baseline_value: float = Form(...),
    accelerations: float = Form(...),
    fetal_movement: float = Form(...),
    uterine_contractions: float = Form(...)
):
    try:
        result = medical_pipeline.process_fetal(baseline_value, accelerations, fetal_movement, uterine_contractions)
        
        action_statements = {
            "Normal": "Excellent baseline heart rate and stable fetal activity recorded. Continue monitoring routine kick-counts.",
            "Suspect": "Atypical cardiotocography (CTG) patterns detected. Rest well, track movements, and repeat the CTG test soon.",
            "Pathological": "Severe fetal distress or oxygen saturation warning. Critical Emergency! Report to the labor room immediately."
        }
        
        return JSONResponse({
            'status': 'success', 
            'result': result,
            'action': action_statements.get(result, "Check configuration configuration parameters.")
        })
    except Exception as e:
        return JSONResponse({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)