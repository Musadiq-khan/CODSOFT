# Task 1: Movie Genre Prediction using NLP

## 🎯 Project Goal
The objective of this project is to build a machine learning classifier that can accurately predict the genre of a movie based solely on its textual plot summary.

## 🛠️ Methodology
1.  **Feature Extraction:** **TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert the raw text plot summaries into a numerical feature vector, capturing the importance of words.
2.  **Text Preprocessing:** Includes lowercasing, removal of special characters, and **stop word removal** (using NLTK) for cleaner input.
3.  **Model:** A **Multinomial Logistic Regression** classifier, chosen for its strong performance and interpretability on sparse text data (like TF-IDF vectors).

## 📂 Project Structure