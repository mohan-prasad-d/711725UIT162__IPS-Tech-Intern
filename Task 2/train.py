import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Reproducibility
np.random.seed(42)

# ==========================================
# 1. DATASET GENERATION (Continuous Boundaries)
# ==========================================
def generate_balanced_dataset(num_samples=4000):
    n_per_class = num_samples // 2
    
    # Pass features (Normal Distribution with distinct means)
    pass_study = np.random.normal(loc=7.5, scale=1.2, size=n_per_class).clip(1.0, 10.0)
    pass_attendance = np.random.normal(loc=88.0, scale=6.0, size=n_per_class).clip(50.0, 100.0)
    pass_marks = np.random.normal(loc=82.0, scale=8.0, size=n_per_class).clip(35.0, 100.0)
    pass_assignments = np.random.normal(loc=85.0, scale=7.0, size=n_per_class).clip(40.0, 100.0)
    
    # Fail features
    fail_study = np.random.normal(loc=3.2, scale=1.1, size=n_per_class).clip(1.0, 10.0)
    fail_attendance = np.random.normal(loc=62.0, scale=8.0, size=n_per_class).clip(50.0, 100.0)
    fail_marks = np.random.normal(loc=45.0, scale=8.0, size=n_per_class).clip(35.0, 100.0)
    fail_assignments = np.random.normal(loc=50.0, scale=8.0, size=n_per_class).clip(40.0, 100.0)
    
    df = pd.DataFrame({
        'roll_no': [f"STU{1001 + i}" for i in range(num_samples)],
        'study_hours': np.round(np.concatenate([pass_study, fail_study]), 1),
        'attendance': np.round(np.concatenate([pass_attendance, fail_attendance]), 1),
        'previous_marks': np.round(np.concatenate([pass_marks, fail_marks]), 1),
        'assignment_scores': np.round(np.concatenate([pass_assignments, fail_assignments]), 1),
        'pass_fail': np.concatenate([np.ones(n_per_class, dtype=int), np.zeros(n_per_class, dtype=int)])
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

df = generate_balanced_dataset(4000)
df.to_csv('balanced_student_data_4000.csv', index=False)

# ==========================================
# 2. TRAIN / VAL / TEST SPLIT (80 / 10 / 10)
# ==========================================
feature_cols = ['study_hours', 'attendance', 'previous_marks', 'assignment_scores']
X = df[feature_cols].values
Y = df[['pass_fail']].values

num_samples = len(df)
indices = np.random.permutation(num_samples)

train_end = int(0.80 * num_samples)  # 3200 samples
val_end = int(0.90 * num_samples)    # 400 val samples, 400 test samples

train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

# Standardization using Training statistics ONLY
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0) + 1e-8

X_train_scaled = (X_train - X_mean) / X_std
X_val_scaled = (X_val - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# 3. ACTIVATION & LOSS FUNCTIONS 

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float) # Explicit float type conversion

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def binary_cross_entropy(Y, A2):
    m = Y.shape[0]
    eps = 1e-15
    A2 = np.clip(A2, eps, 1 - eps)
    return - (1 / m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))

# 4. NEURAL NETWORK WITH ADVANCED FEATURES 


class ProductionNeuralNetwork:
    def __init__(self, input_size=4, hidden_size=8, output_size=1, learning_rate=0.1, lr_decay=0.0005, clip_value=1.0):
        self.initial_lr = learning_rate
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.clip_value = clip_value
        
        # He / Xavier Initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.best_W1, self.best_b1 = None, None
        self.best_W2, self.best_b2 = None, None

    def forward(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = sigmoid(Z2)
        return Z1, A1, Z2, A2

    def backward(self, X, Y, Z1, A1, Z2, A2):
        m = X.shape[0]
        
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, self.W2.T) * relu_derivative(Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Gradient Clipping 
        dW1 = np.clip(dW1, -self.clip_value, self.clip_value)
        db1 = np.clip(db1, -self.clip_value, self.clip_value)
        dW2 = np.clip(dW2, -self.clip_value, self.clip_value)
        db2 = np.clip(db2, -self.clip_value, self.clip_value)
        
        return dW1, db1, dW2, db2

    def train(self, X_train, Y_train, X_val, Y_val, max_epochs=2000, patience=50):
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, max_epochs + 1):
            # LR Scheduler: Time-based Decay 
            self.lr = self.initial_lr / (1.0 + self.lr_decay * epoch)
            
            # Forward & Backward
            Z1, A1, Z2, A2 = self.forward(X_train)
            loss_train = binary_cross_entropy(Y_train, A2)
            
            dW1, db1, dW2, db2 = self.backward(X_train, Y_train, Z1, A1, Z2, A2)
            
            # Update Parameters
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            
            # Validation Step
            _, _, _, A2_val = self.forward(X_val)
            loss_val = binary_cross_entropy(Y_val, A2_val)
            
            train_losses.append(loss_train)
            val_losses.append(loss_val)
            
            # Early Stopping Logic 
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                best_epoch = epoch
                patience_counter = 0
                self.best_W1, self.best_b1 = self.W1.copy(), self.b1.copy()
                self.best_W2, self.best_b2 = self.W2.copy(), self.b2.copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early Stopping triggered at Epoch {epoch}. Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}.")
                    break
                    
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Train Loss: {loss_train:.4f} | Val Loss: {loss_val:.4f} | LR: {self.lr:.5f}")

        # Restore Best Weights
        self.W1, self.b1 = self.best_W1, self.best_b1
        self.W2, self.b2 = self.best_W2, self.best_b2
        
        return train_losses, val_losses, best_epoch

    def find_optimal_threshold(self, X_val, Y_val): # Issue 11
        _, _, _, probs = self.forward(X_val)
        best_threshold, best_acc = 0.5, 0.0
        
        for t in np.arange(0.1, 0.95, 0.05):
            acc = np.mean((probs >= t).astype(int) == Y_val)
            if acc > best_acc:
                best_acc = acc
                best_threshold = t
        return float(best_threshold), float(best_acc)

    def predict(self, X, threshold=0.5):
        _, _, _, probs = self.forward(X)
        return (probs >= threshold).astype(int), probs

# 5. EXECUTION & EVALUATION


nn = ProductionNeuralNetwork(input_size=4, hidden_size=8, output_size=1, learning_rate=0.1, lr_decay=0.0005)
train_losses, val_losses, best_epoch = nn.train(X_train_scaled, Y_train, X_val_scaled, Y_val, max_epochs=2000, patience=50)

# Optimal Threshold Tuning on Validation Set 
optimal_threshold, val_acc = nn.find_optimal_threshold(X_val_scaled, Y_val)
print(f"\nOptimal Decision Threshold tuned on Val Set: {optimal_threshold:.2f} (Val Accuracy: {val_acc*100:.2f}%)")

# Final Evaluation on Test Set
test_preds, test_probs = nn.predict(X_test_scaled, threshold=optimal_threshold)
test_accuracy = float(np.mean(test_preds == Y_test) * 100)
final_test_loss = float(binary_cross_entropy(Y_test, test_probs))

print(f"Final Test Accuracy: {test_accuracy:.2f}% | Final Test Loss: {final_test_loss:.4f}\n")


# 6. SAVE COMPLETE METADATA (NPZ & JSON) 


training_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# NPZ Binary Save
np.savez(
    'model_weights.npz',
    W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2,
    X_mean=X_mean, X_std=X_std,
    learning_rate=nn.initial_lr,
    epochs_trained=len(train_losses),
    best_epoch=best_epoch,
    optimal_threshold=optimal_threshold,
    test_accuracy=test_accuracy,
    training_date=training_date_str
)

# JSON Metadata Save
metadata_json = {
    "model_architecture": {"input_size": 4, "hidden_size": 8, "output_size": 1},
    "training_metadata": {
        "training_date": training_date_str,
        "initial_learning_rate": nn.initial_lr,
        "lr_decay": nn.lr_decay,
        "epochs_trained": len(train_losses),
        "best_epoch": best_epoch,
        "optimal_threshold": optimal_threshold,
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "final_test_loss": final_test_loss,
        "test_accuracy_pct": test_accuracy
    },
    "preprocessing": {
        "feature_mean": X_mean.tolist(),
        "feature_std": X_std.tolist()
    },
    "history": {
        "train_loss": [float(x) for x in train_losses],
        "val_loss": [float(x) for x in val_losses]
    },
    "weights": {
        "W1": nn.W1.tolist(), "b1": nn.b1.tolist(),
        "W2": nn.W2.tolist(), "b2": nn.b2.tolist()
    }
}

with open('model_weights.json', 'w') as f:
    json.dump(metadata_json, f, indent=4)

print("Complete model metadata saved to 'model_weights.npz' and 'model_weights.json'.")

# 7. SAMPLE DEMO FROM ACTUAL TEST SET 

print("\n--- Sampling 5 Random Records from Actual Unseen Test Set ---")
sample_indices = np.random.choice(len(X_test), size=5, replace=False)

sample_df = pd.DataFrame({
    'Study Hours': X_test[sample_indices, 0],
    'Attendance %': X_test[sample_indices, 1],
    'Prev Marks': X_test[sample_indices, 2],
    'Assignment': X_test[sample_indices, 3],
    'Actual': np.where(Y_test[sample_indices].flatten() == 1, 'PASS', 'FAIL'),
    'Predicted': np.where(test_preds[sample_indices].flatten() == 1, 'PASS', 'FAIL'),
    'Probability': np.round(test_probs[sample_indices].flatten(), 4)
})
print(sample_df.to_string(index=False))

# Loss Plot
plt.figure(figsize=(9, 5))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange', linestyle='--')
plt.axvline(best_epoch - 1, color='red', linestyle=':', label=f'Best Epoch ({best_epoch})')
plt.title("Train vs Validation Loss with Early Stopping")
plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()