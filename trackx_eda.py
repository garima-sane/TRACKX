# ==========================================================
# TRACKX 2.0
# Spotify Exploratory Data Analysis
# Internship Project
# ==========================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# Display Settings
# ----------------------------------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("data/processed/spotify_cleaned.csv")

print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ==========================================================
# Investigation 1 : Popularity Distribution
# ==========================================================

print("\n========== POPULARITY STATISTICS ==========\n")

print(df["popularity"].describe())

print(f"\nMean Popularity   : {df['popularity'].mean():.2f}")
print(f"Median Popularity : {df['popularity'].median():.2f}")
print(f"Minimum Popularity: {df['popularity'].min()}")
print(f"Maximum Popularity: {df['popularity'].max()}")
print(f"Std Deviation     : {df['popularity'].std():.2f}")

plt.figure(figsize=(10,6))

sns.histplot(
    data=df,
    x="popularity",
    bins=30,
    kde=True
)

plt.title("Distribution of Spotify Song Popularity")
plt.xlabel("Popularity")
plt.ylabel("Number of Songs")

plt.tight_layout()

plt.savefig("outputs/figures/popularity_distribution.png")

plt.show()


# ----------------------------------------------------------
# Songs with popularity >= 80
# ----------------------------------------------------------

popular_songs = df[df["popularity"] >= 80]

print("\nHit Songs (Popularity ≥ 80)")
print("--------------------------------")
print("Songs :", len(popular_songs))
print("Percentage :", round(len(popular_songs) / len(df) * 100, 2), "%")


# ==========================================================
# Investigation 2 : Feature Correlation Analysis
# ==========================================================

numeric_df = df.select_dtypes(include=["number"])

correlation_matrix = numeric_df.corr()

print("\n========== CORRELATION WITH POPULARITY ==========\n")

print(
    correlation_matrix["popularity"]
    .sort_values(ascending=False)
)

plt.figure(figsize=(12,8))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    annot=False,
    linewidths=0.5
)

plt.title("Correlation Heatmap of Spotify Features")

plt.tight_layout()

plt.savefig("outputs/figures/correlation_heatmap.png")

plt.show()


# ==========================================================
# Investigation 3 : Genre Popularity Analysis
# ==========================================================

genre_popularity = (
    df.groupby("track_genre")["popularity"]
      .mean()
      .sort_values(ascending=False)
)

print("\n========== TOP GENRES BY AVERAGE POPULARITY ==========\n")

print(genre_popularity)

genre_counts = df["track_genre"].value_counts()

print("\nTracks per Genre:\n")
print(genre_counts)

top10_genres = genre_popularity.head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top10_genres.values,
    y=top10_genres.index
)

plt.title("Top 10 Genres by Average Popularity")
plt.xlabel("Average Popularity")
plt.ylabel("Genre")

plt.tight_layout()

plt.savefig("outputs/figures/top10_genres.png")

plt.show()


# ==========================================================
# Investigation 4 : Hit Song Feature Analysis
# ==========================================================

# Create Hit Song Label

df["hit_song"] = np.where(df["popularity"] >= 80, 1, 0)

print("\n========== HIT SONG COUNTS ==========\n")

print(df["hit_song"].value_counts())


# Compare Audio Features

audio_features = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness",
    "duration_ms"
]

hit_song_features = (
    df.groupby("hit_song")[audio_features]
      .mean()
)

print("\n========== AVERAGE AUDIO FEATURES ==========\n")

print(hit_song_features.to_string())


# Difference Analysis

difference = (
    hit_song_features.loc[1]
    -
    hit_song_features.loc[0]
)

print("\n========== FEATURE DIFFERENCE (Hit - Non-Hit) ==========\n")

print(difference)


# Difference Visualization

plt.figure(figsize=(10,6))

difference.plot(
    kind="barh",
    color="steelblue"
)

plt.axvline(
    x=0,
    color="black",
    linewidth=1
)

plt.title("Difference in Audio Features (Hit vs Non-Hit)")
plt.xlabel("Average Difference")
plt.ylabel("Audio Feature")

plt.tight_layout()

plt.savefig("outputs/figures/hit_song_feature_difference.png")

plt.show()


print("\nTRACKX 2.0 Exploratory Data Analysis Completed Successfully.")