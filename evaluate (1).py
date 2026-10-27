import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    average_precision_score
)

def evaluate_model(model, dataloaders, class_names, device, output_name="final"):
    print("\nStarting Evaluation...")
    model.eval()
    
    y_true = []
    y_pred = []
    y_probs = [] # We need probabilities for ROC-AUC and PR-AUC

    # Ensure output directory exists
    os.makedirs('outputs', exist_ok=True)

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Get probabilities using softmax (specifically for the positive class: Pneumonia)
            probs = F.softmax(outputs, dim=1)[:, 1]

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # 1. Calculate Standard Metrics (Precision, Recall, F1, Accuracy)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    # 2. Calculate Advanced Metrics (ROC-AUC & PR-AUC)
    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)

    # 3. Create the Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'{output_name.capitalize()} Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Save the Confusion Matrix Image
    cm_path = f'outputs/{output_name}_confusion_matrix.png'
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.close()

    # 4. Generate and Save the Professional Metrics Report
    report_content = f"""====================================================
EVALUATION REPORT: {output_name.upper()} MODEL
====================================================

1. ADVANCED CLINICAL METRICS (Threshold-Independent)
----------------------------------------------------
ROC-AUC Score : {roc_auc:.4f}
PR-AUC Score  : {pr_auc:.4f}

2. STANDARD METRICS (Threshold-Dependent)
----------------------------------------------------
{report}

3. CONFUSION MATRIX SUMMARY
----------------------------------------------------
True Negatives (Normal correctly identified)   : {cm[0][0]}
False Positives (Normal misdiagnosed as Pneu.) : {cm[0][1]}
False Negatives (Pneumonia missed)             : {cm[1][0]}
True Positives (Pneumonia correctly identified): {cm[1][1]}
====================================================
"""
    
    report_path = f'outputs/{output_name}_metrics_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_content)

    print(report_content)
    print(f"\n[SUCCESS] Visuals saved to: {cm_path}")
    print(f"[SUCCESS] Report saved to : {report_path}")