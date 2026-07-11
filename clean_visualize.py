import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# TRACKX - Data Cleaning & Visualization
# =====================================================

print("=" * 60)
print("TRACKX - Data Cleaning & Visualization")
print("=" * 60)

# Create folders if they don't exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

df = pd.read_csv("data/raw/spotify.csv")

print("\nDataset Loaded Successfully!")
print(f"Original Shape: {df.shape}")

# -----------------------------------------------------
# Remove unnecessary column
# -----------------------------------------------------

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
    print("Removed 'Unnamed: 0' column.")

# -----------------------------------------------------
# Handle Missing Values
# -----------------------------------------------------

rows_before = len(df)

df.dropna(
    subset=["artists", "album_name", "track_name"],
    inplace=True
)

rows_after = len(df)

print(f"Rows removed because of missing values: {rows_before - rows_after}")

# -----------------------------------------------------
# Remove Duplicates
# -----------------------------------------------------

duplicates = df.duplicated().sum()

if duplicates > 0:
    df.drop_duplicates(inplace=True)

print(f"Duplicate rows removed: {duplicates}")

# -----------------------------------------------------
# Handle Outliers (IQR Capping)
# -----------------------------------------------------

print("\nHandling Outliers...")

columns = [
    "duration_ms",
    "tempo",
    "loudness",
    "popularity"
]

for col in columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((df[col] < lower) | (df[col] > upper)).sum()

    print(f"{col}: {outliers} outliers")

    df[col] = df[col].clip(lower=lower, upper=upper)

print("Outliers handled successfully.")

# -----------------------------------------------------
# Save Clean Dataset
# -----------------------------------------------------

df.to_csv(
    "data/processed/spotify_cleaned.csv",
    index=False
)

print("\nClean dataset saved successfully.")

# -----------------------------------------------------
# Visualizations
# -----------------------------------------------------

print("\nGenerating Visualizations...")

# Popularity Histogram
plt.figure(figsize=(8,5))
plt.hist(df["popularity"], bins=20)
plt.title("Popularity Distribution")
plt.xlabel("Popularity")
plt.ylabel("Number of Songs")
plt.tight_layout()
plt.savefig("outputs/popularity_distribution.png")
plt.close()

# Top 10 Genres
plt.figure(figsize=(10,5))
df["track_genre"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Genres")
plt.xlabel("Genre")
plt.ylabel("Number of Tracks")
plt.tight_layout()
plt.savefig("outputs/top_10_genres.png")
plt.close()

# Danceability vs Energy
plt.figure(figsize=(7,6))
plt.scatter(
    df["danceability"],
    df["energy"],
    alpha=0.4
)
plt.title("Danceability vs Energy")
plt.xlabel("Danceability")
plt.ylabel("Energy")
plt.tight_layout()
plt.savefig("outputs/danceability_vs_energy.png")
plt.close()

# Tempo Distribution
plt.figure(figsize=(8,5))
plt.hist(df["tempo"], bins=20)
plt.title("Tempo Distribution")
plt.xlabel("Tempo (BPM)")
plt.ylabel("Number of Songs")
plt.tight_layout()
plt.savefig("outputs/tempo_distribution.png")
plt.close()

# -----------------------------------------------------
# Final Summary
# -----------------------------------------------------

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print(f"Final Dataset Shape : {df.shape}")
print(f"Remaining Missing Values : {df.isnull().sum().sum()}")
print(f"Remaining Duplicates : {df.duplicated().sum()}")

print("\nProject Completed Successfully!")