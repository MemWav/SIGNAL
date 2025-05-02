# test.py

import os
import json
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from models.transformer_only import LandmarkTransformer
from dataset import GestureDatasetCSV  # CustomDataset 대신 GestureDataset 사용
from config import GESTURE

# --- Evaluation ---
class TestConfig:
    def __init__(self):
        self.window_size = 30
        self.batch_size = 64
        self.dataset_dir = './data/csv/test_data'  # JSON folder
        self.model_path = './checkpoint/best_model.pth'  # state_dict path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path, device):
    model = LandmarkTransformer()  # hyperparams must match training
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, test_loader, device):
    all_preds, all_labels = [], []
    total_latency, n_samples = 0.0, 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            start = time.time()
            logits = model(x_batch)
            end = time.time()
            total_latency += (end - start) * 1000
            n_samples += x_batch.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = y_batch.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    # metrics
    if len(all_labels) == 0:
        raise ValueError("No test samples found.")
    num_classes = model.fc.out_features
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    fp_rates = []
    for i in range(num_classes):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp_rates.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    avg_latency = total_latency / n_samples if n_samples > 0 else 0
    return {
        'accuracy': acc,
        'f1_score': f1,
        'confusion_matrix': cm,
        'false_positive_rates': fp_rates,
        'average_latency_ms': avg_latency
    }


def print_results(results, gesture_names=None):
    print("\n=== Model Evaluation Results ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Average Latency: {results['average_latency_ms']:.2f} ms per sample")
    print("\nFalse Positive Rates per class:")
    num_classes = results['confusion_matrix'].shape[0]
    for i in range(num_classes):
        name = gesture_names[i] if gesture_names else i
        print(f"{name}: {results['false_positive_rates'][i]:.4f}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])


def main():
    config = TestConfig()
    dataset = GestureDatasetCSV(config.dataset_dir, window_size=config.window_size)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    model = load_model(config.model_path, config.device)
    results = evaluate_model(model, loader, config.device)
    print_results(results)

if __name__ == '__main__':
    main()
