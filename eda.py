# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Initial Inspection
print("First 5 rows of the training data:")
print(train_data.head())
print("\nData Info:")
print(train_data.info())
print("\nStatistical Summary:")
print(train_data.describe())

# Checking for Missing Values
print("\nMissing Values in Training Data:")
print(train_data.isnull().sum())

# Handle missing values without inplace
train_data['location'] = train_data['location'].fillna('Unknown')
train_data['keyword'] = train_data['keyword'].fillna('Unknown')

# Data Visualization
# Distribution of target variable
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=train_data)
plt.title('Distribution of Target Variable')
plt.xlabel('Target (0 = Non-Disaster, 1 = Disaster)')
plt.ylabel('Count')
plt.show()

# Feature Analysis: Length of Text
train_data['text_length'] = train_data['text'].apply(len)
plt.figure(figsize=(10,6))
sns.histplot(train_data['text_length'], bins=30, kde=True)
plt.title('Distribution of Tweet Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# Outlier Analysis: Boxplot of Text Length by Target
plt.figure(figsize=(8,6))
sns.boxplot(x='target', y='text_length', data=train_data)
plt.title('Boxplot of Text Length by Target')
plt.xlabel('Target (0 = Non-Disaster, 1 = Disaster)')
plt.ylabel('Text Length')
plt.show()

# Text Preprocessing Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize
    words = word_tokenize(text.lower())
    # Remove stop words and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Applying Text Preprocessing
train_data['cleaned_text'] = train_data['text'].apply(preprocess_text)

# Feature Engineering: Adding New Features
train_data['num_hashtags'] = train_data['text'].apply(lambda x: x.count('#'))
train_data['num_mentions'] = train_data['text'].apply(lambda x: x.count('@'))

# Explore Relationships
# Boxplot of the number of hashtags by target
plt.figure(figsize=(8,6))
sns.boxplot(x='target', y='num_hashtags', data=train_data)
plt.title('Boxplot of Number of Hashtags by Target')
plt.xlabel('Target (0 = Non-Disaster, 1 = Disaster)')
plt.ylabel('Number of Hashtags')
plt.show()

# Correlation Analysis
correlation_matrix = train_data[['text_length', 'num_hashtags', 'num_mentions', 'target']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Print final processed data sample
print("Sample of processed data:")
print(train_data[['text', 'cleaned_text', 'text_length', 'num_hashtags', 'num_mentions', 'target']].head())

# End of EDA Script
print("\nExploratory Data Analysis (EDA) Completed.")

