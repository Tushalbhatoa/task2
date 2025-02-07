---

# **Sentiment Analysis on Amazon Reviews**  
This project aims to classify Amazon product reviews as **positive** or **negative** using machine learning. The approach includes **text preprocessing, feature extraction (TF-IDF), and classification using Logistic Regression**.

![Sentiment Analysis](https://upload.wikimedia.org/wikipedia/commons/2/2d/Sentiment_analysis_overview.png)

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Manual Validation](#manual-validation)
- [Challenges and Insights](#challenges-and-insights)
- [Future Improvements](#future-improvements)
- [How to Run the Code](#how-to-run-the-code)
- [Contributing](#contributing)
- [License](#license)

---

## **Project Overview**
Amazon product reviews are a rich source of customer feedback. This project applies **Natural Language Processing (NLP)** techniques to analyze and classify these reviews into positive or negative sentiments.

**Key Steps:**
1. **Preprocess text** (remove stopwords, punctuation, and numbers).
2. **Convert text to numerical features** using **TF-IDF**.
3. **Train a classification model** (Logistic Regression).
4. **Evaluate model performance** using accuracy, precision, and recall.
5. **Validate results** with a manually curated sample.

---

## **Dataset**
We use the **Amazon review dataset** for training and testing. The dataset contains:
- **Text reviews**: Customers' opinions on products.
- **Sentiment labels**: 1 for positive and 0 for negative reviews.

🔗 **Dataset Source:** [Amazon Reviews on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)  

---

## **Installation**
### **1. Clone the repository**
```bash
git clone https://github.com/yourusername/sentiment-analysis-amazon.git
cd sentiment-analysis-amazon
```

### **2. Install dependencies**
Create a virtual environment (optional but recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

Install required libraries:
```bash
pip install -r requirements.txt
```

### **3. Download the dataset**
Ensure that the Amazon review dataset (`amazon_reviews.csv`) is present in the project directory.

---

## **Data Preprocessing**
Before training the model, we **clean the text data**:
✔ Convert text to **lowercase**  
✔ Remove **punctuation and numbers**  
✔ Remove **stopwords** (e.g., "the", "and")  
✔ Tokenization  

### **Preprocessing Code**
```python
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = text.split()  # Tokenization
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens)

# Load dataset
df = pd.read_csv("amazon_reviews.csv")
df['cleaned_review'] = df['review'].apply(preprocess_text)
```

---

## **Feature Extraction**
We use **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical vectors.

### **Why TF-IDF?**
- It gives **higher importance** to words that appear frequently in a document but not across all documents.
- It **filters out common words** that don't add much meaning.

### **TF-IDF Code**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']
```

---

## **Model Training**
We use **Logistic Regression** as the classifier.

### **Train the Model**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
```

---

## **Evaluation**
We evaluate the model using **accuracy, precision, and recall**.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Predictions
y_pred = model.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
```

### **Sample Output**
| Metric     | Score  |
|------------|--------|
| Accuracy   | 89.2%  |
| Precision  | 91.5%  |
| Recall     | 88.7%  |

---

## **Manual Validation**
To verify real-world performance, we manually check predictions on sample reviews.

```python
manual_reviews = [
    "This product is amazing, I love it!",  # Expected: Positive
    "Worst purchase ever. Do not buy!",  # Expected: Negative
    "Decent quality, but not worth the price.",  # Expected: Neutral
]

# Preprocess and predict
manual_reviews_cleaned = [preprocess_text(review) for review in manual_reviews]
manual_reviews_tfidf = vectorizer.transform(manual_reviews_cleaned)
manual_predictions = model.predict(manual_reviews_tfidf)

print(f"Predictions: {manual_predictions}")
```

---

## **Challenges and Insights**
### **Challenges**
- **Handling neutral reviews**: The model is binary (positive/negative), so it struggles with neutral sentiments.
- **Imbalanced dataset**: Some categories of reviews may have more positive than negative examples.
- **Short reviews**: Reviews like "good" or "bad" can be ambiguous.

### **Insights**
- **TF-IDF effectively captures key words**, improving model accuracy.
- **Increasing feature count** (from `max_features=5000` to `10000`) can enhance performance.

---

## **Future Improvements**
- **Try different models**: SVM, Naïve Bayes, or deep learning (LSTMs, BERT).
- **Include neutral reviews**: Convert the model into a **3-class classifier**.
- **Improve dataset balance**: Use techniques like **undersampling or oversampling**.

---

## **How to Run the Code**
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-amazon.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Python script:
   ```bash
   python sentiment_analysis.py
   ```
4. For Jupyter Notebook users:
   ```bash
   jupyter notebook
   ```
   Open `Sentiment_Analysis.ipynb` and run all cells.

---

## **Contributing**
Contributions are welcome! Feel free to open an issue or submit a pull request.

---
