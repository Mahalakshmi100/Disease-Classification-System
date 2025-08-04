# 🧠 Disease Classification System

A Machine Learning-based web application that predicts diseases based on user-input symptoms. This project includes two implementations: one using **FastAPI** and the other using **Flask**.

---

## 📌 Table of Contents

- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Proposed Solution](#proposed-solution)
- [Architecture](#architecture)
- [Tech Stack Used](#tech-stack-used)
- [Installation Steps](#installation-steps)
- [How to Run](#how-to-run)
- [Demo Link](#demo-link)
- [Model Overview](#model-overview)
- [Features](#features)
- [Sample Use-Case](#sample-use-case)
- [Future Scope](#future-scope)
- [Acknowledgements](#acknowledgements)
- [File Structure](#file-structure)

---

## Problem Statement

In the modern healthcare system, it can be difficult to diagnose diseases early without proper tools and resources. Patients may describe symptoms vaguely or partially, making accurate diagnosis a challenge. There's a need for an intelligent system that takes multiple symptoms as input and predicts the most probable disease based on historical data.

---

## Objectives

- Build a predictive model using Machine Learning algorithms.
- Accept up to 10–17 symptoms as user input.
- Predict the most probable disease.
- Provide a user-friendly interface for interaction.
- Implement both **Flask** and **FastAPI** as deployment options.

---

## Proposed Solution

1. Preprocess the dataset by encoding categorical values.
2. Train the model using either:
   - Logistic Regression (Flask)
   - Random Forest Classifier (FastAPI)
3. Save the trained model using `pickle`.
4. Use a web framework to take user input and display the prediction.
5. Deploy the model locally using Flask/FastAPI and HTML forms.

---

## Architecture

The architecture of the Disease Classification System is designed to be modular and scalable. At the frontend, a user-friendly HTML form collects multiple symptoms as input from the user. These inputs are sent to the backend, which can be either a Flask or FastAPI server, depending on the implementation used. The backend handles preprocessing tasks such as encoding the symptoms into a machine-readable format using techniques like Label Encoding or OneHotEncoding. Once preprocessed, the input is passed to a trained Machine Learning model—Logistic Regression in the Flask version and Random Forest in the FastAPI version—which predicts the most likely disease based on the input symptoms. The prediction is then returned to the frontend and displayed in a clean and readable format. This entire system is powered by Python, scikit-learn, and pickle files used to store the trained models and encoders, making it efficient and ready for quick deployment.

---

## Tech Stack Used

| Component       | Technology               |
|----------------|---------------------------|
| Programming     | Python                    |
| Machine Learning| scikit-learn, pandas      |
| Model Storage   | pickle                    |
| Web Framework   | Flask & FastAPI           |
| Frontend        | HTML                      |
| Environment     | Virtualenv (optional)     |

---
## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/Mahalakshmi100/Disease-Classification-System.git
cd Disease-Classification-System/Deployment
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Pre-trained Model Files:
The following files are already included in the repository, so you do not need to retrain the model:

model.pkl → Flask API trained model

disease_model.pkl → FastAPI trained model

label_encoder.pkl → Label encoder used for preprocessing in Flask API

---

## How to Run
### Fast API: 
```bash
python -m uvicorn app:app --reload
```
### Flask API:
```bash
python app.py
```

---

## Model Overview
**Dataset:** Symptom-Disease dataset containing multiple symptom columns and one target (Disease).
**Flask:** Trained using Logistic Regression with Label Encoding.
**FastAPI:** Trained using Random Forest with OneHotEncoding.
**Storage:** Trained models saved using pickle for reuse in APIs.

---

## Features
-Accepts multiple symptoms from the user.
-Predicts the disease instantly.
-User-friendly and clean UI using HTML templates.
-Two API versions: Flask (simple) and FastAPI (modern & scalable).

---

## Sample Use-Case
1. A user selects symptoms like:
Symptom_1: Itching
Symptom_2: Fatigue
...
2. Backend processes the input and makes predictions.
3. Output: Predicted disease such as Fungal Infection, Diabetes, etc.

---

## Output Screenshot

### Flask API  
-**Input Interface**


<img width="477" height="916" alt="image" src="https://github.com/user-attachments/assets/ee7db750-2441-40d4-81aa-1430d2649956" />

-**Prediction Output**


<img width="885" height="254" alt="image" src="https://github.com/user-attachments/assets/941f94a0-d943-465b-b780-0f38dc0051d0" />

---

## Future Scope
-Add confidence score or percentage with each prediction.
-Suggest treatments or nearby doctors based on the prediction.
-Create user login to track patient history.
-Deploy as a full-stack web or mobile application.

---
 
## Acknowledgements
**Scikit-learn**
**Flask Documentation**
**FastAPI Documentation**
**Kaggle for dataset inspiration**
**Open-source community and Stack Overflow**

---
## File Structure

```

Disease-Classification-System/
└── README.md # Project documentation (this file)
└── Deployment/
      ├── pycache/ # Compiled Python files
      │    ├── app.cpython-312.pyc # Compiled Python file
      ├── templates/ # HTML templates for the UI
      │    ├── form.html # Alternate symptom form (if used)
      │    └── result.html # Displays prediction results
      ├── app.py # Flask app script
      ├── model.py # Script containing model training or logic
      ├── Model.pkl # Trained model used in Fast
      ├── disease_model.pkl # Trained model used in FastAPI
      ├── label_encoder.pkl # Encoder for symptom labels (Flask)
      ├── scaler.pkl # Scaler used in preprocessing (optional)
      ├── Original_Dataset (1).csv # Dataset used for training
      ├── Flask API # Contains Fast API code and assets
           ├── templates/ # HTML templates for the UI
           │    ├── index.html # Home page with symptom form
      ├── app.py # Flask app script
      ├── model (2).pkl # Trained model used in Flask
└── README.md # Project documentation (this file)

```

---
