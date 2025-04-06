import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# === 1. Load CSV ===
df = pd.read_csv("TOTALmatchdata.csv")  # Replace with actual file name

# === 2. Split features and label ===
X = df.drop(columns=['label'])
y = df['label']  # 1 = Blue win, 0 = Red win

# === 3. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 4. Train model ===
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

# === 5. Evaluate ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === 6. Save model ===
with open("xgboost_lol_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as 'xgboost_lol_model.pkl'")

# === 7. Plot feature importances ===
plt.figure(figsize=(12, 6))
plt.barh(X.columns, model.feature_importances_)
plt.title("XGBoost Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
