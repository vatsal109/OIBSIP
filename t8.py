import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('google_play_store_data.csv')

# Data Preparation
df.dropna(inplace=True)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

# Category Exploration
category_counts = df['Category'].value_counts()
print(category_counts)

# Metrics Analysis
print(df.groupby('Category')['Rating'].mean().sort_values(ascending=False))
print(df.groupby('Category')['Size'].mean().sort_values(ascending=False))
print(df.groupby('Category')['Installs'].sum().sort_values(ascending=False))

# Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
df['Sentiment'] = df['Reviews'].apply(lambda x: sia.polarity_scores(x)['compound'])
print(df.groupby('Category')['Sentiment'].mean().sort_values(ascending=False))

# Interactive Visualization
sns.set()
plt.figure(figsize=(10,6))
sns.countplot(x='Category', data=df)
plt.title('App Distribution Across Categories')
plt.xticks(rotation=90)
plt.show()