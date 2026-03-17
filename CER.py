# Install dependencies if you haven't
# pip install pandas python-Levenshtein openpyxl

import pandas as pd
import Levenshtein
import os

# Get the folder where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build path to Excel file
file_path = os.path.join(script_dir, "CER.xlsx")

df = pd.read_excel(file_path)

print(df.head())

# Clean whitespace
df['Ground_Truth'] = df['Ground_Truth'].astype(str).str.strip()
df['OCR_Output'] = df['OCR_Output'].astype(str).str.strip()

# Function to calculate Character Error Rate (CER)
def cer(gt, pred):
    return Levenshtein.distance(gt, pred) / max(len(gt), 1)

# Apply CER per row
df['CER'] = df.apply(lambda x: cer(x['Ground_Truth'], x['OCR_Output']), axis=1)

# Compute average CER across all rows
avg_cer = df['CER'].mean()
print("Average CER:", avg_cer)

# Optional: save result with CER column
output_path = os.path.join(script_dir, "cer_results.xlsx")
df.to_excel(output_path, index=False)

print("Saved to:", output_path)
