import os
import tkinter as tk
from tkinter import messagebox
import numpy as np

# 1. LOAD NPZ MODEL WEIGHTS & PARAMETERS

WEIGHTS_FILE = "model_weights.npz"


def load_npz_model():
    if not os.path.exists(WEIGHTS_FILE):
        messagebox.showerror(
            "Error",
            f"'{WEIGHTS_FILE}' file not found! Please run the training script first to generate the '.npz' file.",
        )
        return None
    try:
        # Load npz binary file using standard NumPy
        data = np.load(WEIGHTS_FILE, allow_pickle=True)
        return data
    except Exception as e:
        messagebox.showerror("Error", f"Error loading weights: {str(e)}")
        return None


weights_data = load_npz_model()

if weights_data is not None:
    W1 = weights_data["W1"]
    b1 = weights_data["b1"]
    W2 = weights_data["W2"]
    b2 = weights_data["b2"]

    X_mean = weights_data["X_mean"]
    X_std = weights_data["X_std"]

    # Metadata extracted if present, else defaults applied
    threshold = (
        float(weights_data["optimal_threshold"])
        if "optimal_threshold" in weights_data
        else 0.5
    )
    test_acc = (
        float(weights_data["test_accuracy"])
        if "test_accuracy" in weights_data
        else 0.0
    )

# 2. NUMPY FORWARD PASS LOGIC


def relu(Z):
    return np.maximum(0, Z)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def predict_student_status(study_hours, attendance, prev_marks, assignments):
    X_raw = np.array(
        [[study_hours, attendance, prev_marks, assignments]], dtype=np.float64
    )

    # Manual Feature Standardization
    X_scaled = (X_raw - X_mean) / X_std

    # Forward Propagation using NumPy Matrix Multiplication
    Z1 = np.dot(X_scaled, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    probability = float(A2[0][0])
    status = "PASS" if probability >= threshold else "FAIL"
    return status, probability


# 3. MODERN TKINTER GUI DESIGN


class StudentPredictorApp:

    def __init__(self, root):
        self.root = root
        self.root.title("IPS Tech Intern 2026 - Pure NumPy Neural Network")
        self.root.geometry("480x620")
        self.root.resizable(False, False)

        # Color Palette (Dark Theme)
        self.BG_COLOR = "#181825"
        self.CARD_BG = "#1e1e2e"
        self.ACCENT = "#89b4fa"
        self.TEXT_COLOR = "#cdd6f4"
        self.PASS_COLOR = "#a6e3a1"
        self.FAIL_COLOR = "#f38ba8"

        self.root.configure(bg=self.BG_COLOR)
        self.create_widgets()

    def create_widgets(self):
        # Header Title
        header_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        header_frame.pack(fill="x", pady=20, padx=20)

        title_label = tk.Label(
            header_frame,
            text="Student Performance Predictor",
            font=("Segoe UI", 16, "bold"),
            bg=self.BG_COLOR,
            fg=self.ACCENT,
        )
        title_label.pack()

        acc_text = (
            f"Accuracy: {test_acc:.1f}%"
            if test_acc > 0
            else "Loaded from model_weights.npz"
        )
        subtitle_label = tk.Label(
            header_frame,
            text=f"Pure NumPy Neural Network | {acc_text}",
            font=("Segoe UI", 9),
            bg=self.BG_COLOR,
            fg="#a6adc8",
        )
        subtitle_label.pack(pady=2)

        # Form Card
        form_card = tk.Frame(
            self.root,
            bg=self.CARD_BG,
            bd=0,
            highlightthickness=1,
            highlightbackground="#313244",
        )
        form_card.pack(fill="x", padx=25, pady=5, ipady=10)

        self.entries = {}
        inputs_info = [
            ("Study Hours (per day):", "study_hours"),
            ("Attendance (%):", "attendance"),
            ("Previous Exam Marks (%):", "prev_marks"),
            ("Assignment Scores (%):", "assignments"),
        ]

        for idx, (label_text, key) in enumerate(inputs_info):
            lbl = tk.Label(
                form_card,
                text=label_text,
                font=("Segoe UI", 10, "bold"),
                bg=self.CARD_BG,
                fg=self.TEXT_COLOR,
                anchor="w",
            )
            lbl.pack(
                fill="x", padx=20, pady=(10 if idx == 0 else 5, 2)
            )

            ent = tk.Entry(
                form_card,
                font=("Segoe UI", 11),
                bg="#313244",
                fg="#ffffff",
                insertbackground="#ffffff",
                bd=0,
                relief="flat",
            )
            ent.pack(fill="x", padx=20, ipady=6)
            self.entries[key] = ent

        # Predict Button
        btn = tk.Button(
            self.root,
            text="PREDICT RESULT",
            font=("Segoe UI", 11, "bold"),
            bg=self.ACCENT,
            fg="#11111b",
            activebackground="#b4befe",
            activeforeground="#11111b",
            bd=0,
            cursor="hand2",
            command=self.on_predict,
        )
        btn.pack(fill="x", padx=25, pady=20, ipady=8)

        # Output Display Card
        self.result_card = tk.Frame(
            self.root,
            bg=self.CARD_BG,
            highlightthickness=1,
            highlightbackground="#313244",
        )
        self.result_card.pack(fill="both", expand=True, padx=25, pady=(0, 25))

        self.res_status_lbl = tk.Label(
            self.result_card,
            text="READY FOR INPUT",
            font=("Segoe UI", 16, "bold"),
            bg=self.CARD_BG,
            fg="#6c7086",
        )
        self.res_status_lbl.pack(pady=(20, 5))

        self.res_prob_lbl = tk.Label(
            self.result_card,
            text="Enter student values above and click Predict",
            font=("Segoe UI", 10),
            bg=self.CARD_BG,
            fg="#a6adc8",
        )
        self.res_prob_lbl.pack(pady=2)

    def on_predict(self):
        if weights_data is None:
            messagebox.showerror(
                "Error", "Model weights were not properly loaded."
            )
            return

        try:
            sh = float(self.entries["study_hours"].get())
            att = float(self.entries["attendance"].get())
            pm = float(self.entries["prev_marks"].get())
            asgn = float(self.entries["assignments"].get())

            if not (
                0 <= sh <= 24
                and 0 <= att <= 100
                and 0 <= pm <= 100
                and 0 <= asgn <= 100
            ):
                messagebox.showwarning(
                    "Invalid Range",
                    "Please enter valid ranges (Hours: 0-24, Percentages: 0-100).",
                )
                return

            status, prob = predict_student_status(sh, att, pm, asgn)

            if status == "PASS":
                color = self.PASS_COLOR
                status_text = "🎉 PASS"
            else:
                color = self.FAIL_COLOR
                status_text = "❌ FAIL"

            self.res_status_lbl.config(text=status_text, fg=color)
            self.res_prob_lbl.config(
                text=f"Pass Probability: {prob * 100:.2f}%  |  Threshold: {threshold:.2f}",
                fg=self.TEXT_COLOR,
            )

        except ValueError:
            messagebox.showerror(
                "Input Error", "Please enter numeric values only."
            )


# 4. MAIN EXECUTION

if __name__ == "__main__":
    root = tk.Tk()
    app = StudentPredictorApp(root)
    root.mainloop()