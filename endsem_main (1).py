import torch
import torch.nn as nn
import torch.optim as optim
import data_prep
import model as model_builder
import train
import evaluate

def main():
    print("Initializing End-Semester Project Run...")
    data_dir = 'data'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("\nLoading Data...")
    # FIX: Only unpack the 3 values that get_datasets actually returns
    dataloaders, image_datasets, class_names = data_prep.get_dataloaders(data_dir)
    
    print("Calculating Inverse Frequency Class Weights...")
    # Calculate weights based on Kermany training set counts: Normal=1341, Pneumonia=3875
    count_normal = 1341
    count_pneumonia = 3875
    total_samples = count_normal + count_pneumonia
    
    # Formula: Total / (Number of Classes * Class Count)
    weight_normal = total_samples / (2.0 * count_normal)
    weight_pneumonia = total_samples / (2.0 * count_pneumonia)
    
    class_weights = torch.tensor([weight_normal, weight_pneumonia], dtype=torch.float).to(device)
    print(f"Applied Weights -> Normal: {weight_normal:.4f}, Pneumonia: {weight_pneumonia:.4f}")
    
    print("\nBuilding Imbalance-Aware ResNet18 Architecture...")
    weighted_model = model_builder.build_model(num_classes=len(class_names))
    weighted_model = weighted_model.to(device)

    # ==========================================
    # EXPERIMENT 2: IMBALANCE-AWARE FIX
    # ==========================================
    print("\nSTARTING IMBALANCE-AWARE MODEL TRAINING")
    
    # This is the magic line that fixes the Accuracy Paradox
    weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
    weighted_optimizer = optim.Adam(weighted_model.parameters(), lr=0.001)

    trained_weighted, _ = train.train_model(
        model=weighted_model,
        dataloaders=dataloaders,
        criterion=weighted_criterion,
        optimizer=weighted_optimizer,
        num_epochs=15,
        device=device
    )

    # ==========================================
    # FINAL EVALUATION
    # ==========================================
    evaluate.evaluate_model(
        model=trained_weighted,
        dataloaders=dataloaders,
        class_names=class_names,
        device=device,
        output_name="imbalance_aware"
    )

if __name__ == '__main__':
    main()