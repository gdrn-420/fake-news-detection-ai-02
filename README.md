# 📰 Fake News Detection AI

A Class 12 Artificial Intelligence project that uses Machine Learning and Natural Language Processing to classify news text as FAKE or REAL based on learned text patterns.

## How it works

1. **User Input:** The user enters a news headline or paragraph.
2. **TF-IDF Vectorizer:** Converts text into numerical features.
3. **Logistic Regression:** Analyses the features and predicts a label.
4. **Prediction:** Displays the predicted label and estimated confidence.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- Streamlit

## Project Features

- Simple web interface.
- Text-based news classification.
- Displays prediction confidence.
- Online deployment using Streamlit Community Cloud.

## How to Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The trained model file `fake_news_pipeline.joblib` must be present in the project root folder.

## Model Evaluation

- Dataset size: 200 examples.
- Held-out test accuracy: 60%.
- Grouped cross-validation accuracy: 82.5% ± 12.5%.

These results are from a small, fictional teaching dataset and do not represent real-world fake-news detection accuracy.

## Limitations

- The model learns patterns from its training examples.
- It does not verify facts or check live news sources.
- A REAL prediction does not guarantee that a story is true.
- Important news should always be checked using reliable sources.

## Project Purpose

This project demonstrates how machine learning can be used for text classification through an interactive web application.

**Subject:** Artificial Intelligence (AI 843)  
**Project:** Fake News Detection
