import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('dataset.csv')

# Preprocess text data
df['text'] = df['text'].apply(word_tokenize)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train SVM model
svm = SVC(kernel='linear', C=1)
svm.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = svm.predict(vectorizer.transform(X_test))
print('Accuracy:', accuracy_score(y_test, y_pred))