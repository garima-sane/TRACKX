import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('data/processed/spotify_cleaned.csv')

print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ==========================================================
# Investigation 1: Popularity Distribution
# ==========================================================

print("\nPopularity Statistics\n")

print(df["popularity"].describe())

print(f"\nMean Popularity   : {df['popularity'].mean():.2f}")
print(f"Median Popularity : {df['popularity'].median():.2f}")
print(f"Minimum Popularity: {df['popularity'].min()}")
print(f"Maximum Popularity: {df['popularity'].max()}")
print(f"Std Deviation     : {df['popularity'].std():.2f}")

#Setting up the figures
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

#plt.show()


# How many songs are really popular?
popular_songs = df[df["popularity"] >= 80]

print("Songs with popularity ≥ 80:", len(popular_songs))

print("Percentage:", round(len(popular_songs) / len(df) * 100, 2), "%")

# ==========================================================
# Investigation 2: Correlation Analysis
# ==========================================================

numeric_df = df.select_dtypes(include=["number"])
print("\nNumerical Features:")
print(numeric_df.columns)

correlation_matrix = numeric_df.corr()
print(correlation_matrix["popularity"].sort_values(ascending=False))

# ==========================================================
# Correlation Heatmap
# ==========================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Spotify Audio Features")
plt.tight_layout()
plt.savefig("outputs/figures/correlation_heatmap.png")
#plt.show()

# ==========================================================
# Investigation 3: Top 10 Genres by Average Popularity
# ==========================================================

# Calculate average popularity for each genre
genre_popularity = (
    df.groupby("track_genre")["popularity"]
      .mean()
)

# Sort from highest to lowest and keep only the Top 10
top10_genres = genre_popularity.sort_values(ascending=False).head(10)

# Reverse so the highest value appears at the TOP
top10_genres = top10_genres[::-1]

# Create figure
plt.figure(figsize=(12, 6))

# Horizontal bar chart
sns.barplot(
    x=top10_genres.values,
    y=top10_genres.index
)

# Titles and labels
plt.title("Top 10 Spotify Genres by Average Popularity")
plt.xlabel("Average Popularity")
plt.ylabel("Genre")

plt.tight_layout()
plt.savefig("outputs/figures/top10_genres_popularity.png")
plt.show()