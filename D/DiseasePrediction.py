import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import matplotlib.pyplot as plt
# You may need to install waitress if you plan to deploy the model
# pip install waitress


# --- Data Loading and Preprocessing ---
# Load dataset (example: Pima Diabetes)
df = pd.read_csv("diabetes.csv")

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (critical for SVM/LR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- Model Training and Evaluation ---
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

results = {}
for name, model in models.items():
    # Use scaled data for SVM/LR
    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        # ROC AUC needs probabilities, scale X_test for LR and SVM
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    # Store results
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba)
    }

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
print("Model Evaluation Results:")
print(results_df)


# --- Hyperparameter Tuning (XGBoost Example) ---
print("\nPerforming Hyperparameter Tuning for XGBoost...")
params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}

grid = GridSearchCV(XGBClassifier(), params, cv=5)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")


# --- Feature Importance (XGBoost Example) ---
print("\nAnalyzing Feature Importance for XGBoost...")
xgb = XGBClassifier().fit(X_train, y_train)
plt.figure(figsize=(10, 6))
plt.barh(X.columns, xgb.feature_importances_)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# --- Model Saving ---
print("\nSaving the best model...")
# Train the best model (XGBoost with best params)
best_model = XGBClassifier(max_depth=grid.best_params_['max_depth'],
                           n_estimators=grid.best_params_['n_estimators'])
best_model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Model saved to model.pkl")


# --- Model Loading and Prediction ---
print("\nLoading the saved model and making a prediction...")
# Load the saved model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

print("Model loaded successfully.")

# Create a sample new data point (replace with your actual data)
# The order of features should be the same as the training data
new_data = [[2, 100, 70, 30, 0, 25.0, 0.5, 30]] # Example data

# Convert to DataFrame for consistent processing
new_data_df = pd.DataFrame(new_data, columns=X_train.columns)

# Scale the new data using the same scaler used for training
# Note: Even though XGBoost doesn't require scaling for training,
# the loaded model expects scaled data if it was trained on scaled data.
# However, the original XGBoost model was trained on unscaled data in this notebook,
# so we will use the unscaled new data for prediction with the loaded model.
# If you were to use a scaled model (like LR or SVM), you would scale new_data_df
# using the fitted scaler.

# In this specific case, since the best model (XGBoost) was trained on unscaled data,
# we will predict using the unscaled new data.
prediction = loaded_model.predict(new_data_df)


# Print the prediction
if prediction[0] == 1:
    print("Prediction: Likely to have diabetes")
else:
    print("Prediction: Unlikely to have diabetes")

# You can also get the probability of the prediction
prediction_proba = loaded_model.predict_proba(new_data_df)
print(f"Probability of not having diabetes: {prediction_proba[0][0]:.2f}")
print(f"Probability of having diabetes: {prediction_proba[0][1]:.2f}")