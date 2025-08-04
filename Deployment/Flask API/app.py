import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load dataset
dataset_path = r'C:\Users\Mahalakshmi\OneDrive\Desktop\Flask API\Original_Dataset (1).csv'
data = pd.read_csv(dataset_path)

# Initialize label encoder for symptoms and disease
label_encoder = LabelEncoder()

# Encode symptoms columns
for col in data.columns[1:]:
    data[col] = label_encoder.fit_transform(data[col])

# Separate features and target
X = data.drop('Disease', axis=1)  # Features are all symptom columns
y = data['Disease']  # Target variable is the Disease column

# Encode target variable (Disease)
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Define the prediction API
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form
        symptom_values = [int(request.form[f'Symptom_{i}']) for i in range(1, 18)]

        # Make prediction
        prediction = model.predict([symptom_values])

        # Convert the numeric prediction back to the original disease label
        disease = label_encoder.inverse_transform(prediction)

        return render_template('index.html', prediction=disease[0])

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
