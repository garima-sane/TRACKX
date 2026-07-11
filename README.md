# TRACKX - Spotify Data Cleaning & Visualization

## Overview

# Overview

TRACKX began as a college project focused on exploring and cleaning Spotify music data. As I continued learning data science, I decided to revisit and expand the project as part of my Data Science Internship.

Rather than treating it as a one-time academic assignment, I used this opportunity to refine the project by applying better data preprocessing techniques, improving the project structure, handling data quality issues, and creating meaningful visualizations.

This project reflects my learning journey—from building a basic academic project to developing a more structured data science workflow that mirrors real-world practices.

---

## Objective

The primary objectives of this project are to:

- Clean and preprocess a raw dataset
- Handle missing values
- Detect and handle outliers
- Check for duplicate records
- Generate meaningful visualizations
- Save a cleaned dataset for future analysis

---

## Dataset

- **Dataset:** Spotify Tracks Dataset
- **Records (Original):** 114,000
- **Features (Original):** 21

The dataset contains metadata and audio characteristics of Spotify tracks, including:

- Track Name
- Artist
- Album
- Popularity
- Duration
- Danceability
- Energy
- Loudness
- Tempo
- Genre
- and several other audio features.

---

## Technologies Used

- Python 3
- Pandas
- Matplotlib

---

## Project Structure

```text
TRACKX/
│
├── data/
│   ├── raw/
│   │   └── spotify.csv
│   │
│   └── processed/
│       └── spotify_cleaned.csv
│
├── outputs/
│   ├── popularity_distribution.png
│   ├── top_10_genres.png
│   ├── danceability_vs_energy.png
│   └── tempo_distribution.png
│
├── explore.py
├── clean_visualize.py
└── README.md
```

---

## Workflow

### 1. Data Exploration

The raw dataset was explored to understand:

- Dataset dimensions
- Feature names
- Data types
- Summary statistics
- Missing values
- Duplicate records

---

### 2. Data Cleaning

The cleaning process included:

- Removing unnecessary index columns
- Removing rows containing missing values
- Checking for duplicate records
- Handling outliers using the Interquartile Range (IQR) method
- Saving the cleaned dataset

---

### 3. Data Visualization

The following visualizations were generated:

- Popularity Distribution
- Top 10 Music Genres
- Danceability vs Energy Scatter Plot
- Tempo Distribution

All visualizations are stored in the `outputs` folder.

---

## Results

### Original Dataset

- Records: **114,000**
- Features: **21**

### Cleaned Dataset

- Records: **113,549**
- Features: **20**

### Data Quality

- Missing Values: **0**
- Duplicate Rows: **0**

---

## How to Run

### Explore the dataset

```bash
python explore.py
```

### Clean the dataset and generate visualizations

```bash
python clean_visualize.py
```

---

## Learning Outcomes

## Learning Journey

This project represents an important step in my data science journey.

Although TRACKX started as a college project, revisiting it during my internship allowed me to understand the complete data science workflow in much greater depth. Instead of simply writing code to clean a dataset, I focused on understanding why each preprocessing step is necessary and how it affects the quality of the data before analysis.

Through this project, I strengthened my understanding of:

- Data exploration and quality assessment
- Handling missing values and duplicate records
- Detecting and treating outliers using the IQR method
- Creating meaningful visualizations with Matplotlib
- Organizing a data science project using a structured workflow
- Using Git to manage project versions and improvements

This project marks the transition from completing an academic assignment to approaching data science projects with a more practical and professional mindset.

---

## Future Improvements

Possible enhancements include:

- Interactive dashboards using Plotly or Power BI
- Correlation heatmaps
- Genre-wise audio feature analysis
- Machine learning models for popularity prediction
- Automated data quality reports

---

## Author

**Garima Kushwaha**

B.Tech Computer Science (Data Science)

Data Science Internship Project