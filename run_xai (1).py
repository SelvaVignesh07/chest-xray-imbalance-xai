import torch
import os
import glob
from torchvision import transforms
import model as model_builder
import gradcam

def main():
    print("Initializing Explainable AI (Grad-CAM) Phase...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = ['NORMAL', 'PNEUMONIA']
    
    # Define the exact evaluation transform your Grad-CAM file expects
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\nLoading Trained Imbalance-Aware Model...")
    model = model_builder.build_model(num_classes=2)
    
    # Load your best trained weights
    model_path = 'best_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Successfully loaded best_model.pth!")
    else:
        print(f"ERROR: {model_path} not found.")
        return
        
    model = model.to(device)
    model.eval()

    # Automatically find ResNet18's last convolutional layer
    # Grab the final convolutional block explicitly
    if hasattr(model, 'resnet'):
        target_layer = model.resnet.layer4[-1]
    else:
        target_layer = model.layer4[-1]

    # Grab one Pneumonia and one Normal test image
    print("\nLocating sample test images...")
    try:
        pneu_img = glob.glob('data/test/PNEUMONIA/*.jpeg')[0]
        norm_img = glob.glob('data/test/NORMAL/*.jpeg')[0]
    except IndexError:
        print("ERROR: Could not find images in data/test/. Please check your dataset path.")
        return

    # Run the exact function from YOUR gradcam.py file
    os.makedirs('outputs', exist_ok=True)
    
    print("\nGenerating Heatmap for Pneumonia Patient...")
    gradcam.generate_and_save_gradcam(
        model=model, 
        target_layer=target_layer, 
        image_path=pneu_img, 
        eval_transform=eval_transform,
        device=device, 
        target_class=1,  # 1 is Pneumonia
        class_names=class_names, 
        save_path='outputs/xai_pneumonia.png'
    )

    print("Generating Heatmap for Healthy Patient...")
    gradcam.generate_and_save_gradcam(
        model=model, 
        target_layer=target_layer, 
        image_path=norm_img, 
        eval_transform=eval_transform,
        device=device, 
        target_class=0,  # 0 is Normal
        class_names=class_names, 
        save_path='outputs/xai_normal.png'
    )
    
    print("\nALL GRAD-CAM HEATMAPS GENERATED SUCCESSFULLY! Check your outputs/ folder.")

if __name__ == '__main__':
    main()