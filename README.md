# 🔍 Sentiment Analysis Tool

A full-stack NLP project that classifies customer feedback into three distinct phases: **Positive**, **Neutral**, or **Negative**. Built with classical Machine Learning (Logistic Regression + TF-IDF) and deployed via an interactive Streamlit web dashboard.

---

## 📁 Project Structure

```text
sentiment_analysis/
├── data/
│   ├── generate_dataset.py     # Custom synthetic dataset generator script
│   └── sentiment_data.csv      # Balanced dataset (~2,400–3,000 samples)
├── app/
│   ├── preprocessor.py         # NLTK preprocessing pipeline
│   └── app.py                  # Main Streamlit web application
├── models/
│   ├── sentiment_model.joblib  # Trained model & TF-IDF vectorizer pipeline
│   └── confusion_matrix.png    # Evaluation artefact (Heatmap)
├── train_model.py              # Training script with GridSearchCV
├── requirements.txt            # Python environment dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Key Technologies Used

- **Python**: Core programming language
- **Natural Language Processing (NLP)**: `NLTK` (Tokenization, Stop-words removal, Lemmatization)
- **Feature Extraction**: `Scikit-Learn` (TF-IDF Vectorizer with unigrams & bigrams)
- **Machine Learning**: `Logistic Regression` (GridSearchCV tuned)
- **Web Interface**: `Streamlit` (Interactive multi-tab dashboard)
- **Data & Visualization**: `Pandas`, `Matplotlib`, `Seaborn`

---

## 🚀 Quick Start & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset (Optional)
```bash
python data/generate_dataset.py
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app/app.py
```

---

## 🖥️ Web Application Features

- **🔬 Single Text Analysis**: Live analysis with color-coded feedback alerts (Green for Positive, Red for Negative, Grey for Neutral).
- **📊 Probability Breakdown**: Interactive bar chart visualization for confidence scores across all three classes.
- **📂 Batch Upload**: Process CSV files in bulk with downloadable result predictions.
- **📜 Session History**: Maintains a log of recent predictions made during the live presentation session.
- **📈 Model Metrics**: Dynamic display of accuracy metrics and confusion matrix heatmap.

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 1.00 (100%) |
| Weighted F1-Score | 1.00 |
| Precision / Recall | 1.00 |

*Note: The perfect diagonal score in the confusion matrix demonstrates zero misclassifications on the balanced validation set.*

---

## 👩‍💻 Author
**Dipika Vishwakarma**  
*Minor Project — Internship Presentation*
