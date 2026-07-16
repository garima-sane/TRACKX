# ==========================================================
# TRACKX 3.0
# Spotify Hit Song Prediction using Machine Learning
# ==========================================================

import pandas as pd
import numpy as np

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

