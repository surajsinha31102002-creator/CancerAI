import torch
import joblib
import numpy as np

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the ensemble models
ensemble_models = []
for i in range(1, 6):
    model_path = f"multiclass_ensemble_fold{i}_v3.pth"
    print(f"Loading {model_path}...")
    
    # You need to recreate your model architecture here
    # This is a placeholder - use your actual model class
    from your_api_file import MultiMiRNANet  # Import your model class
    
    model = MultiMiRNANet(input_dim=150, n_classes=5)  # Adjust n_classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    ensemble_models.append(model)

# Create 3 different random inputs
for i in range(3):
    print(f"\n=== Test {i+1} ===")
    X_test = np.random.randn(1, 150) * 2  # 150 features
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        all_probs = []
        for m in ensemble_models:
            out = m(X_t)
            probs = torch.softmax(out, dim=1)
            all_probs.append(probs)
        
        avg_probs = torch.mean(torch.stack(all_probs), dim=0)
        print(f"Average probabilities: {avg_probs.cpu().numpy()}")
        print(f"Max confidence: {avg_probs.max().item() * 100:.2f}%")
        print(f"Predicted class: {torch.argmax(avg_probs, dim=1).item()}")