import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load the data.\venv\Scripts\activate

data = pd.read_csv(r'C:\Users\Mahalakshmi\OneDrive\Desktop\Deployment\Original_Dataset (1).csv')

# Separate features and target variable
X = data.drop('Disease', axis=1)
y = data['Disease']

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Define a column transformer to handle categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'  # This keeps the remaining columns unchanged
)

# Create a pipeline with preprocessing and model training
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=50))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the trained model and categorical columns to a file
with open('disease_model.pkl', 'wb') as model_file:
    pickle.dump((pipeline, categorical_columns), model_file)
