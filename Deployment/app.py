from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd

app = FastAPI()

# Load the trained model and categorical columns
with open('disease_model.pkl', 'rb') as model_file:
    model, categorical_columns = pickle.load(model_file)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, Symptom_1: str = Form(...), Symptom_2: str = Form(...),
                  Symptom_3: str = Form(...), Symptom_4: str = Form(...), Symptom_5: str = Form(...),
                  Symptom_6: str = Form(...), Symptom_7: str = Form(...), Symptom_8: str = Form(...),
                  Symptom_9: str = Form(...), Symptom_10: str = Form(...)):
    data = pd.DataFrame([[Symptom_1, Symptom_2, Symptom_3, Symptom_4, Symptom_5,
                          Symptom_6, Symptom_7, Symptom_8, Symptom_9, Symptom_10]],
                        columns=[f'Symptom_{i}' for i in range(1, 11)])

    # Ensure all categorical columns are present in the input
    for col in categorical_columns:
        if col not in data.columns:
            data[col] = ""

    # Predict the disease
    prediction = model.predict(data)
    return templates.TemplateResponse("result.html", {"request": request, "prediction": prediction[0]})
