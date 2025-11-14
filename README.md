# 📧 SMS Spam Detection using Machine Learning and NLP

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

### 🧹 Text Preprocessing
- Lowercasing
- Removing punctuation
- Removing stopwords
- Lemmatization (optional)

example
```
def lower_replace(series):
    output = series.str.lower()
    output = output.str.replace(r'\[.*?\]', '', regex=True)
    output = output.str.replace(r'[^\w\s]', '', regex=True)
    return output

def lemma_stopwords(text):
    doc = nlp(text)
    output = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(output)

def clean_normalize(text):
    output = lower_replace(text)
    output = output.apply(lemma_stopwords)
    return output
```
```
df['Cleaned_text'] = df['message'].apply(clean_text)
```

--- 

### ✨ TF-IDF Vectorization
```
tf = TfidfVectorizer(stop_words='english',ngram_range=(1,2),min_df=0.01,max_df=0.9)
Xt = tf.fit_transform(train_df['Clean_Message'])
Xt_df = pd.DataFrame(Xt.toarray(),columns=tf.get_feature_names_out())
Xt_df
```

---

### 🤖 Model Training
Models trained:
- Logistic Regression
- Naive Bayes
- Random Forest

---

### 📊 Model Performance
1. Logistic Regression (TF-IDF)

    - Accuracy: 91%
    - Precision:
        - Non-Spam: 98%
        - Spam: 60%
    - Recall:
        - Non-Spam: 92%
        - Spam: 88%

2. Naive Bayes (TF-IDF)

    - Accuracy: 91%
    - Precision:
       - Non-Spam: 99%
       - Spam: 61%
    - Recall:
       - Non-Spam: 91%
       - Spam: 96%

3. Random Forest (TF-IDF)

    - Accuracy: 92%
    - Precision:
        - Non-Spam: 98%
        - Spam: 65%
    - Recall:
        - Non-Spam: 93%
        - Spam: 83%

---

### ✅ Recommended Model for Deployment
Multinomial Naive Bayes
✔ Best recall for Spam (96%)
✔ Lower false negatives
✔ Fast & lightweight

---

### 🚀 Future Improvements
- Try more models
- Deploy using Flask API
- Build a Streamlit web UI

### 👤 Author

**Aldous Dsouza**  
Machine Learning & Data Science Enthusiast  
- 📍 Goa, India  
- 📧 Email: *aldous27d.work@outlook.com*    

Feel free to connect for collaborations or improvements!
