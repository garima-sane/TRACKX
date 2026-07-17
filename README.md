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

TRACKX 3.0 extends the project from exploratory data analysis into predictive machine learning by developing a classification model capable of identifying whether a Spotify track is likely to become a **Hit Song** based solely on its intrinsic audio characteristics. To simulate a real-world prediction scenario, the `popularity` column was intentionally excluded from the training features to prevent target leakage, ensuring that the model learns meaningful musical patterns rather than existing popularity scores.

A binary target variable (`hit_song`) was created using a popularity threshold of **80**, where songs with popularity greater than or equal to 80 were labeled as **Hit (1)** and the remaining songs as **Non-Hit (0)**. The dataset was then divided into **80% training** and **20% testing** sets using a fixed random state (`42`) to ensure reproducibility across experiments.

The model was trained using ten selected audio features:

- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Duration (ms)

The `time_signature` feature was excluded due to its limited contribution during exploratory analysis.

---

## 🧪 Experimental Journey

Instead of immediately selecting a complex model, TRACKX followed an iterative experimental approach by evaluating multiple machine learning algorithms and analyzing their strengths and limitations.

### 1. Logistic Regression (Baseline)

The first baseline model achieved an accuracy of **98.8%**, but failed to identify any hit songs, producing **0% Precision and 0% Recall** for the minority class. This experiment demonstrated that **accuracy alone is an unreliable evaluation metric for highly imbalanced datasets**.

### 2. Balanced Logistic Regression

To address the class imbalance, Logistic Regression was trained using balanced class weights. Recall increased dramatically to **82%**, showing that the model successfully detected most hit songs. However, Precision dropped to **2%**, indicating that a large number of non-hit songs were incorrectly classified as hits. This experiment clearly highlighted the trade-off between **Precision** and **Recall**.

### 3. Decision Tree

A Decision Tree classifier was then implemented to capture non-linear relationships among the audio features. The model achieved **98.9% Accuracy**, **54% Precision**, and **56% Recall**, providing a significantly more balanced performance compared to Logistic Regression while maintaining high overall accuracy.

### 4. Random Forest

Finally, a Random Forest classifier was trained to evaluate whether an ensemble learning approach could further improve prediction quality. The model achieved the best overall performance with **99.0% Accuracy**, **95% Precision**, **56% Recall**, and an **F1-score of 0.71**. Compared with the Decision Tree, the Random Forest dramatically reduced false positive predictions while maintaining the same Recall, making it the final selected model.

---

## 📊 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|---------:|----------:|-------:|---------:|
| Logistic Regression | 98.8% | 0.00 | 0.00 | 0.00 |
| Balanced Logistic Regression | 62.0% | 0.02 | 0.82 | 0.05 |
| Decision Tree | 98.9% | 0.54 | 0.56 | 0.55 |
| **Random Forest** | **99.0%** | **0.95** | **0.56** | **0.71** |

---

## 🌲 Feature Importance

Feature importance analysis revealed that the Random Forest model relied on a combination of multiple audio characteristics rather than a single dominant feature. Tempo, Loudness, Valence, Speechiness, and Acousticness emerged as the most influential predictors, suggesting that hit songs are defined by a combination of musical attributes rather than any individual feature.

---

## 📉 Confusion Matrix Analysis

The final Random Forest model correctly classified **22,440 non-hit songs** and **147 hit songs**, while producing only **7 false positive predictions**. Although **116 hit songs** remained undetected, the model demonstrated excellent reliability when predicting hit songs, achieving **95% Precision** with a balanced Recall of **56%**.

---

## 🎯 Key Findings

- High Accuracy alone does not guarantee an effective classifier for imbalanced datasets.
- Precision, Recall, and F1-score provide a more reliable evaluation than Accuracy alone.
- Class imbalance significantly influences model performance and should always be investigated.
- Ensemble learning (Random Forest) substantially improved prediction quality over a single Decision Tree.
- Hit song prediction depends on a combination of multiple audio characteristics rather than a single dominant feature.
- A systematic experimental approach provided stronger evidence for model selection than relying solely on a single performance metric.

---

## 🏁 Conclusion

TRACKX 3.0 evolved beyond simply training a machine learning model into a systematic investigation of classification performance on an imbalanced Spotify dataset. By progressively evaluating multiple algorithms, analyzing their trade-offs, interpreting feature importance, and validating predictions using a confusion matrix, the project identified **Random Forest** as the most suitable model for hit song prediction. This workflow reflects the practical methodology followed in real-world machine learning projects, where model selection is driven by experimentation, evaluation, and evidence rather than accuracy alone.

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

# 📂 Project Structure

```text
TRACKX/
│
├── .venv/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── .gitignore
├── clean_visualize.py      # TRACKX 1.0 - Data Cleaning & Visualization
├── explore.py              # TRACKX 2.0 - Exploratory Data Analysis
├── trackx_ml.py            # TRACKX 3.0 - Machine Learning & Model Evaluation
│
└── README.md
```

# Current Progress

| Stage | Progress |
|--------|----------|
| Data Cleaning | ✅ Complete |
| Exploratory Data Analysis | ✅ Complete |
| Machine Learning | ✅ Complete |
| Deployment | ⏳ Planned |

# Author

**Garima Kushwaha**

B.Tech Computer Science (Data Science)