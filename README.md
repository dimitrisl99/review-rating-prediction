# 🧠 Review Rating Prediction (NLP Project)

This project focuses on predicting product ratings (1–5 stars) based on review text using Machine Learning and NLP techniques.

---

## 📊 Dataset

- Amazon product reviews (~5k subset)
- Features:
  - `reviewText` (input)
  - `overall` (target rating)

---

## ⚙️ Approach

### 1. Preprocessing
- Removed missing values
- Selected relevant columns
- Converted ratings to integers

### 2. Feature Engineering
- TF-IDF Vectorization
- N-grams (unigrams + bigrams)
- Max features: 5000

### 3. Models
- Logistic Regression (baseline)
- Linear SVM (better performance)

---

## 📈 Results

### 🔵 Linear SVM
- Accuracy: ~0.78
- Strong performance on extreme classes (1⭐, 5⭐)

### 🟢 Logistic Regression
- Accuracy: ~0.71
- More balanced but lower overall performance

---

## 🔍 Key Insights

- Dataset is highly imbalanced (majority = 5⭐)
- Model tends to favor dominant class
- Confusion mainly occurs between adjacent ratings (e.g. 4⭐ vs 5⭐)
- Mixed sentiment reviews are difficult to classify

---

## 📊 Evaluation

- Accuracy
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- Error Analysis (manual inspection of misclassified reviews)

---

## 🧱 Project Structure
```commandline
src/
│── data_loader.py
│── preprocess.py
│── features.py
│── train.py
│── evaluate.py
│── error_analysis.py

main.py
```
---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py