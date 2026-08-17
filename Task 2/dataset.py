import numpy as np
import pandas as pd

np.random.seed(42)
n_per_class = 2000  # 2000 Pass + 2000 Fail = 4000 Students

# Pass class data generation
pass_study = np.round(np.random.uniform(5.0, 10.0, n_per_class), 1)
pass_attendance = np.round(np.random.uniform(75.0, 100.0, n_per_class), 1)
pass_marks = np.round(np.random.uniform(60.0, 100.0, n_per_class), 1)
pass_assignments = np.round(np.random.uniform(65.0, 100.0, n_per_class), 1)
pass_labels = np.ones(n_per_class, dtype=int)  # 1 = Pass

# Fail class data generation
fail_study = np.round(np.random.uniform(1.0, 5.0, n_per_class), 1)
fail_attendance = np.round(np.random.uniform(40.0, 75.0, n_per_class), 1)
fail_marks = np.round(np.random.uniform(20.0, 55.0, n_per_class), 1)
fail_assignments = np.round(np.random.uniform(30.0, 60.0, n_per_class), 1)
fail_labels = np.zeros(n_per_class, dtype=int)  # 0 = Fail

# Concatenate features and labels
study_hours = np.concatenate([pass_study, fail_study])
attendance = np.concatenate([pass_attendance, fail_attendance])
previous_marks = np.concatenate([pass_marks, fail_marks])
assignment_scores = np.concatenate([pass_assignments, fail_assignments])
pass_fail = np.concatenate([pass_labels, fail_labels])

# Create DataFrame
df = pd.DataFrame({
    'study_hours': study_hours,
    'attendance': attendance,
    'previous_marks': previous_marks,
    'assignment_scores': assignment_scores,
    'pass_fail': pass_fail,
})

# Shuffle Data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add Roll Number
df.insert(0, 'roll_no', [f'STU{1001 + i}' for i in range(len(df))])

# Save as CSV
df.to_csv('balanced_student_data_4000.csv', index=False)

print('Dataset created with exact 50-50 balance!')
print('\nPass (1) vs Fail (0) Count:')
print(df['pass_fail'].value_counts())
print('\nFirst 5 Rows:')
print(df.head())