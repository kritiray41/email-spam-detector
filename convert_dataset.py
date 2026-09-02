import pandas as pd

# Read the SMS Spam Collection dataset
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["v1", "v2"],
    encoding="utf-8"
)

# Save as CSV
df.to_csv("data/spam.csv", index=False)

print("Dataset converted successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution:")
print(df["v1"].value_counts())