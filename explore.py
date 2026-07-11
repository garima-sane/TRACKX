import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Makes plots look a bit nicer
plt.style.use("ggplot")

df = pd.read_csv("data/raw/spotify.csv")
print("Shape:", df.shape)
print(df.head())
print(df.columns.tolist())
print(df.info())
print(df.describe(include="all"))
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())