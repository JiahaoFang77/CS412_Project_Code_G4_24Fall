import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
  DistilBertTokenizer, DistilBertForSequenceClassification,
  BertTokenizer, BertForSequenceClassification
)
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Define model configurations
MODEL_CONFIGS = {
  'bert-base': {
    'name': 'bert-base-uncased',
    'tokenizer': BertTokenizer,
    'model': BertForSequenceClassification
  },
  'bert-large': {
    'name': 'bert-large-uncased',
    'tokenizer': BertTokenizer,
    'model': BertForSequenceClassification
  },
  'distilbert': {
    'name': 'distilbert-base-uncased',
    'tokenizer': DistilBertTokenizer,
    'model': DistilBertForSequenceClassification
  }
}

RANDOM_SEEDS = [42, 123, 456]

# Create results directory with subdirectories for each model
def create_directories():
  for model_name in MODEL_CONFIGS.keys():
    for seed in RANDOM_SEEDS:
      os.makedirs(f'bert_results/{model_name}/seed_{seed}', exist_ok=True)

class DisasterTweetsDataset(Dataset):
  def __init__(self, texts, targets=None, tokenizer=None, max_length=160):
    self.texts = texts
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(self.texts.iloc[idx])
    
    encoding = self.tokenizer(
      text,
      add_special_tokens=True,
      max_length=self.max_length,
      truncation=True,
      padding='max_length',
      return_tensors='pt'
    )

    item = {
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten()
    }

    if self.targets is not None:
      item['target'] = torch.tensor(self.targets.iloc[idx], dtype=torch.long)

    return item

def train_model(model, train_loader, val_loader, device, model_name, seed, epochs=2, lr=1e-5):
  output_dir = f'bert_results/{model_name}/seed_{seed}'
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()
  
  best_val_acc = 0
  train_losses = []
  val_accuracies = []
  
  with open(f'{output_dir}/training_log.txt', 'w') as f:
    f.write(f"Training Progress Log for {model_name} (Seed: {seed})\n")
    f.write("="*50 + "\n")
  
  for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
    
    for batch in progress_bar:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      targets = batch['target'].to(device)

      optimizer.zero_grad()
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      loss = criterion(outputs.logits, targets)
      
      loss.backward()
      optimizer.step()
      
      total_loss += loss.item()
      progress_bar.set_postfix({'loss': total_loss / len(train_loader)})

    # Validation
    model.eval()
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
      for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        
        val_predictions.extend(predicted.cpu().numpy())
        val_targets.extend(targets.cpu().numpy())
    
    val_report = classification_report(val_targets, val_predictions)
    val_accuracy = (np.array(val_predictions) == np.array(val_targets)).mean()
    
    with open(f'{output_dir}/training_log.txt', 'a') as f:
      f.write(f"\nEpoch {epoch + 1}/{epochs}\n")
      f.write(f"Training Loss: {total_loss / len(train_loader):.4f}\n")
      f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
      f.write("\nValidation Report:\n")
      f.write(val_report)
      f.write("\n" + "="*50 + "\n")
    
    print(f'Epoch {epoch + 1} - Validation Accuracy: {val_accuracy:.4f}')
    
    val_accuracies.append(val_accuracy)
    train_losses.append(total_loss / len(train_loader))
    
    if val_accuracy > best_val_acc:
      best_val_acc = val_accuracy
      torch.save(model.state_dict(), f'{output_dir}/best_model.pt')
  
  # Plot training curves
  plt.figure(figsize=(12, 5))
  
  plt.subplot(1, 2, 1)
  plt.plot(train_losses)
  plt.title(f'Training Loss ({model_name}, Seed: {seed})')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  
  plt.subplot(1, 2, 2)
  plt.plot(val_accuracies)
  plt.title(f'Validation Accuracy ({model_name}, Seed: {seed})')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  
  plt.tight_layout()
  plt.savefig(f'{output_dir}/training_curves.png')
  plt.close()
  
  return train_losses, val_accuracies

def evaluate_model(model, data_loader, device, model_name, seed, dataset_name=""):
  output_dir = f'bert_results/{model_name}/seed_{seed}'
  model.eval()
  all_targets = []
  all_predictions = []
  all_probabilities = []
  
  with torch.no_grad():
    for batch in data_loader:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      targets = batch['target'].to(device) if 'target' in batch else None
      
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      probabilities = torch.softmax(outputs.logits, dim=1)
      _, predicted = torch.max(outputs.logits, 1)
      
      all_probabilities.extend(probabilities.cpu().numpy())
      all_predictions.extend(predicted.cpu().numpy())
      if targets is not None:
        all_targets.extend(targets.cpu().numpy())
  
  if all_targets:
    report = classification_report(all_targets, all_predictions)
    with open(f'{output_dir}/{dataset_name.lower()}_report.txt', 'w') as f:
      f.write(f"Classification Report for {dataset_name} Set ({model_name}, Seed: {seed}):\n")
      f.write(report)
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_targets, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    f1_score = tp / (tp + ((fn + fp) / 2))
    
    disp = ConfusionMatrixDisplay.from_predictions(
      all_targets,
      all_predictions,
      display_labels=["Not Disaster", "Disaster"],
      cmap=plt.cm.Blues
    )
    
    plt.title(f"Confusion Matrix ({model_name}, Seed: {seed})\nF1 Score: {f1_score:.2f}")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{dataset_name.lower()}_confusion_matrix.png')
    plt.close()
  
  return all_predictions, all_probabilities

def main():
  print("Creating directories...")
  create_directories()
  
  print("Loading data...")
  df_train = pd.read_csv("train.csv")
  df_test = pd.read_csv("test.csv")
  
  # Split data
  X = df_train["text"]
  y = df_train["target"]
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")
  
  # Train and evaluate each model variant with different seeds
  for model_name, config in MODEL_CONFIGS.items():
    print(f"\nTraining {model_name.upper()} variants...")
    
    for seed in RANDOM_SEEDS:
      print(f"\nTraining with seed {seed}")
      torch.manual_seed(seed)
      np.random.seed(seed)
      
      X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
      
      # Initialize tokenizer and model
      tokenizer = config['tokenizer'].from_pretrained(config['name'])
      model = config['model'].from_pretrained(config['name'], num_labels=2)
      model = model.to(device)
      
      # Create datasets
      train_dataset = DisasterTweetsDataset(X_train, y_train, tokenizer)
      val_dataset = DisasterTweetsDataset(X_val, y_val, tokenizer)
      test_dataset = DisasterTweetsDataset(df_test["text"], tokenizer=tokenizer)
      
      # Create dataloaders
      train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
      val_loader = DataLoader(val_dataset, batch_size=32)
      test_loader = DataLoader(test_dataset, batch_size=32)
      
      # Train model
      print(f"\nStarting training for {model_name} (Seed: {seed})...")
      train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, device, model_name, seed
      )
      
      # Load best model and evaluate
      print(f"\nEvaluating best model for {model_name} (Seed: {seed})...")
      model.load_state_dict(torch.load(f'bert_results/{model_name}/seed_{seed}/best_model.pt'))
      
      print("\nEvaluating on training set...")
      evaluate_model(model, train_loader, device, model_name, seed, "Training")
      
      print("\nEvaluating on validation set...")
      evaluate_model(model, val_loader, device, model_name, seed, "Validation")
      
      # Generate predictions for test set
      print("\nGenerating predictions for test set...")
      test_predictions, test_probabilities = evaluate_model(
        model, test_loader, device, model_name, seed, "Test"
      )
      
      # Save test predictions
      submission = pd.DataFrame({
        'id': df_test['id'],
        'target': test_predictions,
        'disaster_probability': [prob[1] for prob in test_probabilities]
      })
      submission.to_csv(f'bert_results/{model_name}/seed_{seed}/submission.csv', index=False)

if __name__ == "__main__":
  main()