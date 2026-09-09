from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from preprocess import clean_text


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==========================================
# 2. RENAME COLUMNS
# ==========================================

df.columns = ["label", "text"]

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 3. CONVERT LABELS TO NUMBERS
# ==========================================

df["label_num"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nLabel distribution:")
print(df["label_num"].value_counts())


# ==========================================
# 4. CLEAN TEXT
# ==========================================

print("\nCleaning messages...")

df["cleaned_text"] = df["text"].apply(clean_text)

print("Text cleaning completed!")


# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["cleaned_text"],
    df["label_num"],
    test_size=0.20,
    random_state=42,
    stratify=df["label_num"]
)

print("\nTrain/Test Split:")
print("Training samples:", len(X_train_raw))
print("Testing samples:", len(X_test_raw))


# ==========================================
# 6. WORD-LEVEL TF-IDF
# ==========================================

print("\nCreating word-level TF-IDF...")

word_tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_word = word_tfidf.fit_transform(X_train_raw)
X_test_word = word_tfidf.transform(X_test_raw)

print("Word TF-IDF features:", X_train_word.shape[1])


# ==========================================
# 7. CHARACTER-LEVEL TF-IDF
# ==========================================

print("\nCreating character-level TF-IDF...")

char_tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=5000,
    sublinear_tf=True,
    min_df=2
)

X_train_char = char_tfidf.fit_transform(X_train_raw)
X_test_char = char_tfidf.transform(X_test_raw)

print("Character TF-IDF features:", X_train_char.shape[1])


# ==========================================
# 8. COMBINE WORD + CHARACTER FEATURES
# ==========================================

print("\nCombining TF-IDF features...")

X_train = hstack([
    X_train_word,
    X_train_char
]).tocsr()

X_test = hstack([
    X_test_word,
    X_test_char
]).tocsr()

print("Combined X_train shape:", X_train.shape)
print("Combined X_test shape:", X_test.shape)


# ==========================================
# 9. CREATE MODELS
# ==========================================

models = {

    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced"
    )
}


# ==========================================
# 10. TRAIN MODELS
# ==========================================

trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    trained_models[name] = model

    print(f"{name} training completed!")


# ==========================================
# 11. MAKE PREDICTIONS
# ==========================================

predictions = {}

for name, model in trained_models.items():

    predictions[name] = model.predict(X_test)

    print(f"{name} predictions completed!")


# ==========================================
# 12. EVALUATE MODELS
# ==========================================

results = {}

print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")

for name, y_pred in predictions.items():

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    print(f"\n{name}")
    print("------------------------------------------")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


# ==========================================
# 13. SELECT BEST MODEL
# ==========================================

best_model_name = max(
    results,
    key=lambda name: results[name]["f1_score"]
)

best_model = trained_models[best_model_name]

print("\nBest model selected:", best_model_name)
print("Selection criterion: F1-score")

print(
    f"Best F1-score: "
    f"{results[best_model_name]['f1_score']:.4f}"
)


# ==========================================
# 14. SAVE MODELS
# ==========================================

joblib.dump(
    best_model,
    "models/spam_model.pkl"
)

joblib.dump(
    trained_models["Linear SVM"],
    "models/svm_model.pkl"
)

joblib.dump(
    trained_models["Logistic Regression"],
    "models/logistic_model.pkl"
)

# Save BOTH vectorizers
joblib.dump(
    word_tfidf,
    "models/word_tfidf_vectorizer.pkl"
)

joblib.dump(
    char_tfidf,
    "models/char_tfidf_vectorizer.pkl"
)

print("\nModels saved successfully!")

print("models/spam_model.pkl")
print("models/svm_model.pkl")
print("models/logistic_model.pkl")

print("\nVectorizers saved successfully!")

print("models/word_tfidf_vectorizer.pkl")
print("models/char_tfidf_vectorizer.pkl")