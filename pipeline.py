import pandas as pd
import numpy as np

class ClinicalInferencePipeline:
    def __init__(self, maternal_model, maternal_scaler, fetal_model, fetal_scaler):
        self.maternal_model = maternal_model
        self.maternal_scaler = maternal_scaler
        self.fetal_model = fetal_model
        self.fetal_scaler = fetal_scaler
        
        # Exact column structures matching original notebooks
        self.maternal_cols = ["Age", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
        self.fetal_cols = [
            'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 
            'light_decelerations', 'severe_decelerations', 'prolongued_decelerations', 
            'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
            'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability', 
            'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
            'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median', 
            'histogram_variance', 'histogram_tendency'
        ]

    def process_maternal(self, age: float, diastolic_bp: float, bs: float, body_temp: float, heart_rate: float):
        # Structuring matrix
        custom_data = pd.DataFrame([[age, diastolic_bp, bs, body_temp, heart_rate]], columns=self.maternal_cols)
        
        # Scaling matrix and mapping names back to remove terminal warnings
        scaled_array = self.maternal_scaler.transform(custom_data)
        scaled_df = pd.DataFrame(scaled_array, columns=self.maternal_cols)
        
        prediction = self.maternal_model.predict(scaled_df)[0]
        labels = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
        return labels[prediction]

    def process_fetal(self, baseline_value: float, accelerations: float, fetal_movement: float, uterine_contractions: float):
        # Setting up baseline array template
        input_dict = {col: 0.0 for col in self.fetal_cols}
        
        # Clinical parameters adjustment for scaling safety bounds
        input_dict['abnormal_short_term_variability'] = 35.0   
        input_dict['mean_value_of_short_term_variability'] = 1.3
        input_dict['histogram_mode'] = baseline_value
        input_dict['histogram_mean'] = baseline_value
        input_dict['histogram_median'] = baseline_value
        
        # Custom parameters injections
        input_dict['baseline value'] = baseline_value
        input_dict['accelerations'] = accelerations
        input_dict['fetal_movement'] = fetal_movement
        input_dict['uterine_contractions'] = uterine_contractions
        
        fetal_data = pd.DataFrame([input_dict], columns=self.fetal_cols)
        
        # Scale & pass feature mappings safely
        scaled_array = self.fetal_scaler.transform(fetal_data)
        scaled_df = pd.DataFrame(scaled_array, columns=self.fetal_cols)
        
        prediction = self.fetal_model.predict(scaled_df)[0]
        labels = {1: "Normal", 2: "Suspect", 3: "Pathological"}
        return labels.get(prediction, str(prediction))