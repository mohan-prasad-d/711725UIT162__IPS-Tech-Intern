# Student Performance Predictor: Production-Grade Neural Network from Scratch

A framework-free Feed-Forward Neural Network implemented entirely in **Pure NumPy**, featuring an interactive **Tkinter GUI** for real-time inference. 

This project accurately classifies student outcomes (`PASS` / `FAIL`) based on key academic performance metrics without relying on high-level deep learning frameworks like PyTorch, TensorFlow, Keras, or Scikit-learn.

---

## Key Features & Production Engineering

* **Zero Framework Dependencies:** Built completely from scratch using Python and NumPy matrix operations.
* **Data Leakage Prevention:** Feature standardization parameters ($\mu, \sigma$) are computed strictly on training data and reused across validation, testing, and GUI inference.
* **Proper Weight Initialization:** Implements **He (Kaiming) Initialization** for the ReLU hidden layer and **Xavier (Glorot) Initialization** for the Sigmoid output layer to ensure gradient stability.
* **Production Optimization:**
  * **Gradient Clipping:** Prevents exploding gradients by capping gradients to $[-1.0, 1.0]$.
  * **Time-based Learning Rate Decay:** Dynamically decays learning rate over epochs to achieve smooth convergence.
  * **Early Stopping:** Tracks validation loss with a patience parameter to restore best weights and prevent overfitting.
  * **Validation Threshold Tuning:** Dynamically computes the optimal decision threshold on the validation set instead of defaulting to $0.5$.
* **Dual Model Serialization:** Exports trained parameters, preprocessing stats, and metadata into both `.npz` (binary) and `.json` (human-readable) formats.
* **Modern Tkinter GUI:** A dark-themed desktop application for real-time model evaluation and interactive prediction.

---

## Dataset

The model processes **4 continuous input features** to predict binary pass/fail classification:

| Feature Name | Description | Value Range |
| :--- | :--- | :--- |
| `study_hours` | Daily study duration in hours | $1.0 - 10.0$ |
| `attendance` | Class attendance percentage | $50.0\% - 100.0\%$ |
| `previous_marks` | Previous exam performance | $35.0\% - 100.0\%$ |
| `assignment_scores` | Average score on assignments | $40.0\% - 100.0\%$ |
| **`pass_fail`** | **Target Label (0 = FAIL, 1 = PASS)** | **Binary** |

The script automatically generates a balanced dataset of 4,000 synthetic records (`balanced_student_data_4000.csv`) with normal distributions for continuous boundary learning.

---

## Network Architecture

```text
[ Input Layer ] (4 Features)
       │
       ▼  (Weights: W1 [4x8], Biases: b1 [1x8] | He Initialization)
[ Hidden Layer ] (8 Neurons + ReLU Activation)
       │
       ▼  (Weights: W2 [8x1], Biases: b2 [1x1] | Xavier Initialization)
[ Output Layer ] (1 Neuron + Sigmoid Activation)
       │
       ▼  (Binary Cross-Entropy Loss)
[ Prediction Output ] (Probability → Thresholding → PASS / FAIL)