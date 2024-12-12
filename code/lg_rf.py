import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import re
from collections import Counter
import scipy.stats as stats
import os

# Define model configurations and random seeds
MODEL_CONFIGS = {
  'logistic_regression': {
    'name': 'Logistic Regression',
    'model': LogisticRegression,
    'params': {'max_iter': 1000}
  },
  'random_forest': {
    'name': 'Random Forest',
    'model': RandomForestClassifier,
    'params': {'n_estimators': 200}
  }
}

RANDOM_SEEDS = [42, 123, 456]

def create_directories():
  """Create directory structure for results"""
  for model_name in MODEL_CONFIGS.keys():
    for seed in RANDOM_SEEDS:
      os.makedirs(f'results/{model_name}/seed_{seed}', exist_ok=True)
  os.makedirs('results/word_analysis', exist_ok=True)

def preprocess_text(text):
  """Basic text preprocessing"""
  # Convert to lowercase
  text = str(text).lower()
  
  # Remove URLs
  text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
  
  # Remove special characters and numbers
  text = re.sub(r'[^\w\s]', '', text)
  text = re.sub(r'\d+', '', text)
  
  # Remove extra whitespace
  text = ' '.join(text.split())
  
  return text

def analyze_word_distributions(df):
  """Analyze word distributions and calculate KL divergence"""
  # Split into disaster and non-disaster tweets
  disaster_tweets = df[df['target'] == 1]['processed_text']
  non_disaster_tweets = df[df['target'] == 0]['processed_text']
  
  # Create word frequency distributions
  vectorizer = CountVectorizer(min_df=5)  # Ignore rare words
  
  # Get word frequencies for both classes
  disaster_counts = vectorizer.fit_transform(disaster_tweets)
  non_disaster_counts = vectorizer.transform(non_disaster_tweets)
  
  # Calculate probabilities
  disaster_probs = disaster_counts.sum(axis=0).A1 / disaster_counts.sum()
  non_disaster_probs = non_disaster_counts.sum(axis=0).A1 / non_disaster_counts.sum()
  
  # Smooth probabilities to avoid zeros
  epsilon = 1e-10
  disaster_probs = disaster_probs + epsilon
  non_disaster_probs = non_disaster_probs + epsilon
  
  # Normalize
  disaster_probs = disaster_probs / disaster_probs.sum()
  non_disaster_probs = non_disaster_probs / non_disaster_probs.sum()
  
  # Calculate KL divergence
  kl_div = stats.entropy(disaster_probs, non_disaster_probs)
  
  # Get vocabulary
  vocab = vectorizer.get_feature_names_out()
  
  # Get top words for each class
  top_disaster_words = pd.DataFrame({
    'word': vocab,
    'frequency': disaster_probs
  }).sort_values('frequency', ascending=False).head(20)
  
  top_non_disaster_words = pd.DataFrame({
    'word': vocab,
    'frequency': non_disaster_probs
  }).sort_values('frequency', ascending=False).head(20)
  
  # Plot word frequencies
  plt.figure(figsize=(15, 6))
  
  plt.subplot(1, 2, 1)
  sns.barplot(data=top_disaster_words, x='frequency', y='word')
  plt.title('Top 20 Words in Disaster Tweets')
  
  plt.subplot(1, 2, 2)
  sns.barplot(data=top_non_disaster_words, x='frequency', y='word')
  plt.title('Top 20 Words in Non-Disaster Tweets')
  
  plt.tight_layout()
  plt.savefig('results/word_analysis/word_distributions.png')
  plt.close()
  
  # Save KL divergence
  with open('results/word_analysis/kl_divergence.txt', 'w') as f:
    f.write(f'KL Divergence between disaster and non-disaster word distributions: {kl_div}')
  
  return kl_div

def train_and_evaluate_model(model_name, model, X_train, X_test, y_train, y_test, seed):
  """Train and evaluate a model with detailed metrics"""
  output_dir = f'results/{model_name.lower().replace(" ", "_")}/seed_{seed}'
  
  pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', model)
  ])
  
  # Train model
  print(f"\nTraining {model_name} (Seed: {seed})...")
  pipeline.fit(X_train, y_train)
  
  # Make predictions
  y_pred = pipeline.predict(X_test)
  
  # Get and save classification report
  report = classification_report(y_test, y_pred)
  print(f"\n{model_name} Classification Report (Seed: {seed}):")
  print(report)
  
  with open(f'{output_dir}/classification_report.txt', 'w') as f:
    f.write(f"Classification Report for {model_name} (Seed: {seed}):\n")
    f.write(report)
  
  # Plot confusion matrix
  plt.figure(figsize=(8, 6))
  ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Not Disaster", "Disaster"],
    cmap=plt.cm.Blues
  )
  plt.title(f"{model_name} Confusion Matrix (Seed: {seed})")
  plt.tight_layout()
  plt.savefig(f'{output_dir}/confusion_matrix.png')
  plt.close()
  
  # For Logistic Regression, analyze feature importance
  if isinstance(model, LogisticRegression):
    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    feature_importance = pd.DataFrame({
      'feature': feature_names,
      'importance': abs(pipeline.named_steps['classifier'].coef_[0])
    })
    
    # Plot top 20 most important features
    plt.figure(figsize=(12, 6))
    top_features = feature_importance.nlargest(20, 'importance')
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'Top 20 Most Important Features (Seed: {seed})')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png')
    plt.close()
    
    # Save feature importance
    feature_importance.sort_values('importance', ascending=False).to_csv(
      f'{output_dir}/feature_importance.csv', index=False
    )
  
  return pipeline

def main():
  print("Creating directories...")
  create_directories()
  
  # Load data
  print("Loading and preprocessing data...")
  df_train = pd.read_csv("train.csv")
  df_test = pd.read_csv("test.csv")
  
  # Preprocess text
  df_train['processed_text'] = df_train['text'].apply(preprocess_text)
  df_test['processed_text'] = df_test['text'].apply(preprocess_text)
  
  # Analyze word distributions
  print("\nAnalyzing word distributions...")
  kl_div = analyze_word_distributions(df_train)
  print(f"KL Divergence: {kl_div}")
  
  # Train and evaluate models with different seeds
  for model_key, config in MODEL_CONFIGS.items():
    print(f"\nTraining {config['name']} variants...")
    
    for seed in RANDOM_SEEDS:
      print(f"\nTraining with seed {seed}")
      np.random.seed(seed)
      
      # Split data
      X = df_train['processed_text']
      y = df_train['target']
      X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
      
      # Initialize model with seed
      model_params = config['params'].copy()
      model_params['random_state'] = seed
      model = config['model'](**model_params)
      
      # Train and evaluate model
      pipeline = train_and_evaluate_model(
        config['name'],
        model,
        X_train,
        X_val,
        y_train,
        y_val,
        seed
      )
      
      # Generate predictions for test set
      print(f"\nGenerating predictions for test set ({config['name']}, Seed: {seed})...")
      predictions = pipeline.predict(df_test['processed_text'])
      submission = pd.DataFrame({
        'id': df_test['id'],
        'target': predictions
      })
      submission.to_csv(f'results/{model_key}/seed_{seed}/submission.csv', index=False)
  
  print("\nAnalysis complete. Results organized in 'results' directory:")
  print("└── results/")
  print("    ├── word_analysis/")
  print("    │   ├── word_distributions.png")
  print("    │   └── kl_divergence.txt")
  print("    ├── logistic_regression/")
  for seed in RANDOM_SEEDS:
    print(f"    │   ├── seed_{seed}/")
    print("    │   │   ├── classification_report.txt")
    print("    │   │   ├── confusion_matrix.png")
    print("    │   │   ├── feature_importance.png")
    print("    │   │   ├── feature_importance.csv")
    print("    │   │   └── submission.csv")
  print("    └── random_forest/")
  for seed in RANDOM_SEEDS:
    print(f"        ├── seed_{seed}/")
    print("        │   ├── classification_report.txt")
    print("        │   ├── confusion_matrix.png")
    print("        │   └── submission.csv")

if __name__ == "__main__":
  main()