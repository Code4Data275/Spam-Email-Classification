# 📧 SMS Spam Detection using Machine Learning

A machine learning project to classify SMS messages as **Spam** or **Non-Spam** using text preprocessing, TF-IDF vectorization, and multiple ML models.

---

## 📌 Project Overview

This project builds a text-classification system to detect spam messages.

The workflow includes:

- Text cleaning & preprocessing  
- TF-IDF feature extraction  
- Training ML models (Logistic Regression, Naive Bayes, Random Forest)  
- Evaluating models  
- Saving & loading the best model  
- Predicting new messages  

---

## 🗂️ Dataset Format

Your dataset must contain two columns:

| label     | message                           |
|-----------|------------------------------------|
| Spam      | "Claim your free prize now!"       |
| Non-Spam  | "Hey, are we meeting today?"       |

Convert labels if needed:

---

## 🔧 Installation

```
bash
pip install numpy pandas scikit-learn joblib
```

---

### Text Preprocessing
- Lowercasing
- Removing punctuation
- Removing stopwords
- Lemmatization (optional)

```
example
df['Cleaned_text'] = df['message'].apply(clean_text)
```

--- 
