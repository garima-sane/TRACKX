# TRACKX

### Data Cleaning, Exploratory Data Analysis and Machine Learning for Spotify Track Popularity Prediction

## Project Overview

TRACKX is a multi-stage data science project developed to study Spotify music data and understand the factors influencing song popularity. The project follows the complete lifecycle of a data science workflow, beginning with raw data preprocessing and gradually progressing toward predictive machine learning models.

Rather than treating data cleaning, exploration and modelling as independent tasks, TRACKX is structured into progressive versions where each stage builds upon the previous one.

The project roadmap is shown below.

| Version | Stage | Status |
|----------|-------------------------------|-----------|
| TRACKX 1.0 | Data Cleaning & Preprocessing | ✅ Completed |
| TRACKX 2.0 | Exploratory Data Analysis | ✅ Completed |
| TRACKX 3.0 | Machine Learning Models | 🔄 Planned |
| TRACKX 4.0 | Deployment & Business Dashboard | 🔄 Planned |

The current repository contains the completed work for **TRACKX 2.0**, while future versions will continue to expand the project.

# TRACKX 1.0 — Data Cleaning & Preprocessing

The first phase focused on transforming the raw Spotify dataset into a reliable dataset suitable for analysis.

Major tasks included:

- Removing duplicate records
- Handling missing values
- Standardizing data types
- Removing inconsistent records
- Creating the cleaned dataset used throughout the project

Output

```
spotify_cleaned.csv
```

# TRACKX 2.0 — Exploratory Data Analysis

The second phase investigates patterns within the cleaned dataset and extracts business insights from Spotify audio features.

The analysis is divided into four investigations.

## Investigation 1
### Popularity Distribution

Objectives

- Understand how popularity is distributed
- Compute descriptive statistics
- Identify highly popular songs

Outputs

- Popularity Distribution Histogram
- Descriptive Statistics

---

## Investigation 2
### Correlation Analysis

Objectives

- Study relationships between numerical audio features
- Identify features associated with popularity

Outputs

- Correlation Matrix
- Correlation Heatmap

---

## Investigation 3
### Genre Analysis

Objectives

- Compare average popularity across genres
- Identify genres with consistently higher popularity

Outputs

- Top 10 Genre Popularity Chart
- Genre Statistics

---

## Investigation 4
### Hit Song Analysis

Songs with popularity greater than or equal to **80** were classified as Hit Songs.

The analysis compares average audio characteristics between hit and non-hit tracks.

Audio features analysed include

- Danceability
- Energy
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Loudness
- Duration

Outputs

- Hit vs Non-Hit Feature Comparison
- Business Insights

# Business Insights

The exploratory analysis produced the following observations.

- Song popularity exhibits a wide distribution with relatively few highly popular tracks.
- Individual audio features have weak correlations with popularity.
- Popularity differs considerably across music genres.
- Hit songs display distinct audio characteristics compared to non-hit songs.
- Song success is better explained through multiple audio features rather than a single attribute.

These findings motivate the next stage of the project involving predictive machine learning.

# TRACKX 3.0 — Machine Learning (Upcoming)

The third phase will focus on predicting whether a Spotify track is likely to become a hit.

Planned work includes

- Feature Engineering
- Model Training
- Model Comparison
- Performance Evaluation
- Hit Song Prediction

# TRACKX 4.0 — Deployment & Dashboard (Upcoming)

The final stage aims to convert the machine learning solution into a deployable application.

Future objectives include

- Interactive Dashboard
- Model Deployment
- Prediction Interface
- Business Analytics Dashboard

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn *(TRACKX 3.0)*

# Repository Structure

```
TRACKX/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   └── figures/
│
├── trackx_clean.py
├── trackx_eda.py
├── requirements.txt
└── README.md
```

# Current Progress

| Stage | Progress |
|--------|----------|
| Data Cleaning | ✅ Complete |
| Exploratory Data Analysis | ✅ Complete |
| Machine Learning | 🔄 In Progress |
| Deployment | ⏳ Planned |

# Author

**Garima Kushwaha**

B.Tech Computer Science (Data Science)