import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt  

# Data
iris = load_iris()
X, y = iris.data, iris.target
custom_names = ['Rose', 'Sunflower', 'Lotus']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest — 100 trees!
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict + Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=custom_names))

# Feature importance
for feat, imp in zip(iris.feature_names, model.feature_importances_):
    print(f"{feat}: {imp:.4f}")

# Figure — 2 with plots 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))


# Plot 1 — Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=custom_names).plot(ax=axes[0])
axes[0].set_title("Confusion Matrix — Random Forest")


# Plot 2 — Feature Importance bar chart
axes[1].bar(iris.feature_names, model.feature_importances_, color=['#E53E3E','#D97706','#16A34A','#7F77DD'])
axes[1].set_title("Feature Importance")
axes[1].set_ylabel("Importance Score")
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig("day5_results.png", dpi=150)
plt.show()
print("Figure saved!")