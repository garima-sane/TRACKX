# ==========================================================
# TRACKX 3.0
# Spotify Hit Song Prediction using Machine Learning
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

# Machine Learning Libraries
from sklearn.model_selection import train_test_split


# Load the cleaned Spotify dataset
df = pd.read_csv('data/processed/spotify_cleaned.csv')

#=================================
#Dataset Validation
#=================================
print("\nDataset Shape:", df.shape)
print("\nDataset Columns:", df.columns)
print("\n first 5 rows of the dataset:\n", df.head())
print("\n Missing Values in the dataset:\n", df.isnull().sum())

audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'valence',
                  'tempo', 'duration_ms']

#Creating Labels and Features

x=df[audio_features]
y=df["hit_song"]
x_train, x_test, y_train, y_test = train_test_split(
                                                    x, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=47)   

print("\nTraining Set Shape:", x_train.shape, y_train.shape)
print("\nTesting Set Shape:", x_test.shape, y_test.shape)

#==============================
#Model Training
#==============================

#logistic regression model (balanced class and baseline class) #
lr_model = LogisticRegression( max_iter=1000)
lr_model.fit(x_train, y_train)
print("\nLogistic Regression Model Training Completed Successfully.")

lrb_model = LogisticRegression(class_weight='balanced', max_iter=1000)
lrb_model.fit(x_train, y_train)
print("\nBalanced Logistic Regression Model Training Completed Successfully.")

# Decision Tree Model #
dt_model = DecisionTreeClassifier(random_state=47)
dt_model.fit(x_train, y_train)
print("\nDecision Tree Model Training Completed Successfully.")

# Random Forest Model #
rf_model = RandomForestClassifier(random_state=47)
rf_model.fit(x_train, y_train)
print("\nRandom Forest Model Training Completed Successfully.")

#==============================
#Model Evaluation
#==============================

lr_pred = lr_model.predict(x_test)
lrb_pred = lrb_model.predict(x_test)
tree_pred = dt_model.predict(x_test)

#==============================
#Model Accuracy, Precision, Recall, F1-Score
#==============================

#Accuracy 
lr_accuracy= accuracy_score(y_test, lr_pred)
print(f"\n Logistic Regression Model Accuracy: {lr_accuracy:.3f}")

lrb_accuracy = accuracy_score(y_test, lrb_pred)
print(f"\n Balanced Logistic Regression Model Accuracy: {lrb_accuracy:.3f}")

tree_accuracy = accuracy_score(y_test, tree_pred)
print(f"\nDecision Tree Model Accuracy: {tree_accuracy:.3f}")

forest_accuracy = accuracy_score(y_test, rf_model.predict(x_test))
print(f"\nRandom Forest Model Accuracy: {forest_accuracy:.3f}")

#Classification Report
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, lr_pred))
print("\nBalanced Logistic Regression Classification Report:\n", classification_report(y_test, lrb_pred))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, tree_pred))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_model.predict(x_test)))

# ==========================================================
# Random Forest Feature Importance
# ==========================================================

feature_importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========\n")

print(feature_importance)
plt.figure(figsize=(10,6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("outputs/feature_importance.png")

#plt.show()

# ==========================================================
# Confusion Matrix for Random Forest Model
# ==========================================================

cm = confusion_matrix(
    y_test,
    rf_model.predict(x_test)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Hit", "Hit"]
)

disp.plot(
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.show()

#==========================================================
# Model Comparison Visualization
#==========================================================
models = [
    "Logistic",
    "Balanced LR",
    "Decision Tree",
    "Random Forest"
]

precision = [0.00, 0.02, 0.54, 0.95]

recall = [0.00, 0.82, 0.56, 0.56]

f1 = [0.00, 0.05, 0.55, 0.71]

X=np.arange(len(models))
width=0.30
plt.xticks (X, models)

plt.figure(figsize=(10,6))

plt.bar(X - width, precision, width, label="Precision")
plt.bar(X, recall, width, label="Recall")
plt.bar(X + width, f1, width, label="F1 Score")

plt.xticks(X, models)
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.xlabel("Models")
plt.title("Comparison of Machine Learning Models")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.show()
print("\nTRACKX 3.0 Machine Learning Model Training and Evaluation Completed Successfully.")
