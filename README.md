# Women Health Care Analysis & Diagnostic Platform

An industry-standard, end-to-end Machine Learning web application designed to predict maternal health risk levels and classify fetal health conditions using clinical data. Built with a decoupled high-performance architecture using **FastAPI** for the backend engine and an asynchronous, responsive frontend UI.

---

## Key Features
- **Dual Inference Engine:** 
  - **Maternal Risk Predictor:** Analyzes patient vitals (Age, BP, Blood Sugar, Temperature, Heart Rate) to classify risk as *Low*, *Mid*, or *High*.
  - **Fetal Health CTG Classifier:** Classifies Cardiotocography metrics into *Normal*, *Suspect*, or *Pathological* health states.
- **Production-Ready Architecture:** Clean separation of concerns with a dedicated `pipeline.py` inference wrapper preventing terminal data alignment warnings.
- **Asynchronous UI:** Dynamic jQuery integrated forms provide seamless, real-time diagnostic outcomes and clinical action guidelines without full-page reloads.
- **Robust Environment:** Built-in isolated virtual environment structure matching software engineering best practices.

---

## Project Architecture & Directory Structure

```text
Women-Health-Care-AI/
│
├── app.py                         # Core FastAPI Web Server 
├── pipeline.py                    # Preprocessing & ML Inference Engine
├── scaler.pkl                     # Pre-trained Maternal Data Scaler
├── maternal_health_gbc_model.pkl   # Gradient Boosting Classifier for Maternal Risk
├── fetal_scaler.pkl               # Pre-trained Fetal Data Scaler
├── fetal_health_model.pkl         # Machine Learning Model for Fetal Health
├── requirements.txt               # Main Dependency Manifest
├── .gitignore                     # Git Tracking Exclusion Configurations
│
└── templates/
    └── index.html                 # Asynchronous Frontend User Interface


Tech Stack & Dependencies

Backend: FastAPI, Uvicorn, Jinja2, Python-Multipart

Machine Learning: Scikit-Learn (v1.3.2), Pandas, NumPy, Joblib

Frontend: HTML5, CSS3, Bootstrap 5, FontAwesome Icons, jQuery (Asynchronous AJAX integration)

1. Clone the repository:
   ```bash
   git clone (https://github.com/abhisheknishad23/Maternal-Health-Risk-levels.git)
   

2. Set Up a Virtual Environment
   python -m venv venv
   venv\Scripts\activate

3. Install Required Dependencies
   pip install -r requirements.txt

4. Run the Platform
   python app.py