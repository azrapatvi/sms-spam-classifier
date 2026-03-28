# 📩 SMS Spam Classifier using Word2Vec

A machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Word2Vec embeddings and multiple ML models.

---

## 🔍 What does this project do?

You give it an SMS message → it tells you if it's **Spam 🚨** or **Ham ✅**

Example:
```
Input:  "Congratulations! You won a free iPhone. Click here to claim now."
Output: SPAM 🚨
```

---

## 📁 Dataset

- **Name:** SMS Spam Collection Dataset
- **Source:** [KrishNaik's GitHub](https://raw.githubusercontent.com/krishnaik06/Complete-Data-Science-With-Machine-Learning-And-NLP-2024/refs/heads/main/26-CompleteNLP%20For%20Machine%20Learning/Practicals/SpamClassifier-master/smsspamcollection/SMSSpamCollection)
- **Size:** 5,572 SMS messages
- **Labels:** `ham` (not spam) and `spam`

---

## 🧠 How it works (Step by Step)

### 1. Text Preprocessing
- Removed special characters and numbers
- Converted text to lowercase
- Applied **Lemmatization** (e.g., "running" → "run")
- Removed **Stopwords** (e.g., "the", "is", "and")

### 2. Word2Vec Embeddings
- Trained a **Word2Vec model** on the cleaned messages
- Each word gets converted into a **100-dimensional vector**
- For each SMS message, calculated the **average of all word vectors** → gives one fixed-size vector per message

### 3. Model Training
Trained and compared multiple models:
| Model | Notes |
|---|---|
| Logistic Regression | Simple baseline |
| Decision Tree | Tree-based |
| Random Forest | Best performer ✅ |
| Gradient Boosting | Ensemble method |
| XGBoost | Boosting method |

### 4. Hyperparameter Tuning
- Used **RandomizedSearchCV** with 5-fold cross-validation on top 3 models
- Final model: **Random Forest** with tuned parameters

### 5. Prediction on New Input
- Preprocess the new message the same way
- Convert to avg Word2Vec vector
- Predict using the trained Random Forest model

---

## 🚀 How to run

### 1. Install dependencies
```bash
pip install gensim nltk scikit-learn xgboost tqdm pandas numpy
```

### 2. Download NLTK data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

### 3. Run the notebook
Open `text_classify.ipynb` in Jupyter and run all cells top to bottom.

### 4. Predict on your own message
At the end of the notebook:
```python
new_message = "You have won a free prize! Call now."
print(predict_message(new_message))
# Output: SPAM 🚨
```

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Gensim** – Word2Vec model
- **NLTK** – Text preprocessing
- **Scikit-learn** – ML models, train-test split, evaluation
- **XGBoost** – Boosting classifier
- **Pandas & NumPy** – Data handling
- **tqdm** – Progress bar

---

## 📊 Project Structure

```
sms-spam-classifier-word2vec/
│
├── text_classify.ipynb   # Main notebook with all code
└── README.md             # This file
```

---

## 💡 Key Concepts Used

- **Word2Vec** – Converts words to dense vectors based on context
- **Average Word2Vec** – Represents a whole sentence as one vector
- **Lemmatization** – Reduces words to their base form
- **RandomizedSearchCV** – Finds best hyperparameters efficiently

---

## 👤 Author

Made by [Your Name] — feel free to fork and improve!
