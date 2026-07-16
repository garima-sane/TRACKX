# ==========================================================
# TRACKX 3.0
# Spotify Hit Song Prediction using Machine Learning
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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