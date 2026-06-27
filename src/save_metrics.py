import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def save_model_metrics(y_test, y_pred, models_dir, reports_dir, model_name="Logistic Regression (Baseline)"):
    """
    Evaluates predictions, saves the metrics to a JSON file, 
    and plots/saves a confusion matrix.
    """
    
    # 1. Calculate final metrics
    test_acc = accuracy_score(y_test, y_pred)
    # Using pos_label='Dementia' assuming binary classification where Dementia is the positive class
    test_f1 = f1_score(y_test, y_pred, pos_label=1)
    
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # 2. Save Metrics to JSON
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    metrics = {
        "model": model_name,
        "test_accuracy": round(test_acc, 4),
        "test_f1_score": round(test_f1, 4)
    }
    
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    # 3. Plot and Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Healthy', 'Dementia'], 
                yticklabels=['Healthy', 'Dementia'])
    
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        
    safe_model_name = model_name.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
    cm_path = os.path.join(reports_dir, f"{safe_model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTask Complete! Saved metrics to {metrics_path} and confusion matrix to {cm_path}")

# Example of how you would call this if running the file directly (for testing):
if __name__ == "__main__":
    print("This is a utility module. Import 'save_model_metrics' to use it in your training scripts.")