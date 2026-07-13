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

plt.show()


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
print(correlation_matrix)

#Heatmap of the correlation matrix
plt.figure(figsize=(12,8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Spotify Features")

plt.tight_layout()

plt.savefig("outputs/figures/correlation_heatmap.png")

plt.show()