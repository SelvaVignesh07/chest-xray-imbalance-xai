# Addressing the Accuracy Paradox in Chest X-Ray Classification using Explainable AI

## Project Overview
This project extends the work of Alsaafin et al. (2024) by evaluating a ResNet18 architecture on the severely imbalanced Kermany Chest X-Ray dataset (74% Pneumonia, 26% Normal). We expose the **Accuracy Paradox**—where a naive baseline model achieves 62.5% overall accuracy but collapses to a 0% recall for the minority healthy class. 

## The Solution: Inverse Frequency Class Weighting
To mathematically prevent majority-class collapse without artificially altering the dataset, we engineered an Imbalance-Aware model using Inverse Frequency Class Weighting integrated directly into the PyTorch `nn.CrossEntropyLoss` function.
* **Minority Weight (Normal):** 1.94
* **Majority Weight (Pneumonia):** 0.67

## Clinical Metrics Achieved
* **Overall Accuracy:** 91.35%
* **ROC-AUC Score:** 0.9639
* **PR-AUC Score:** 0.9747
* **Normal Class Recall:** 81.62% (Successfully recovering the 0% baseline failure)

## Explainable AI (Grad-CAM)
To ensure the model is anchoring its decisions on actual physiological features (lung opacities) rather than dataset artifacts, we implemented a custom Grad-CAM extraction on the final convolutional layer. The heatmaps confirm deep clinical alignment.

---
**Team:** 
* Selva Vignesh V (CB.AI.U4AID25149)
* Viayamurughan S (CB.AI.U4AID25160)
* Mahalakshmi R (CB.AI.U4AID25167)
* Pavani B (CB.AI.U4AID25110)
