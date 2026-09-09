import pandas as pd
from preprocess import clean_text


# Load dataset
df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)

print("========== DATASET INFORMATION ==========")

print("\nShape:")
print(df.shape)

print("\nClass Distribution:")
print(df["v1"].value_counts())

print("\n========== PREPROCESSING TEST ==========")

# Clean the first message
original = df["v2"].iloc[0]
cleaned = clean_text(original)

print("\nOriginal message:")
print(original)

print("\nCleaned message:")
print(cleaned)