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

audio_features = ['danceability', 'energy', 'loudness', , 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'valence',
                  'tempo']

