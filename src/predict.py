import joblib
from preprocess import clean_text
from scipy.sparse import hstack


# ==========================================
# 1. LOAD MODELS
# ==========================================

best_model = joblib.load("models/spam_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
lr_model = joblib.load("models/logistic_model.pkl")

# Load both TF-IDF vectorizers
word_tfidf = joblib.load("models/word_tfidf_vectorizer.pkl")
char_tfidf = joblib.load("models/char_tfidf_vectorizer.pkl")


# ==========================================
# 2. TAKE EMAIL INPUT
# ==========================================

print("\nEnter your email.")
print("Type END on a separate line when finished.\n")

lines = []

while True:
    line = input()

    if line.strip() == "END":
        break

    lines.append(line)

email = "\n".join(lines)


# ==========================================
# 3. CLEAN EMAIL
# ==========================================

cleaned_email = clean_text(email)

print("\nCleaned text:")
print(cleaned_email)


# ==========================================
# 4. CREATE WORD + CHARACTER TF-IDF
# ==========================================

word_vector = word_tfidf.transform([cleaned_email])
char_vector = char_tfidf.transform([cleaned_email])

message_vector = hstack([
    word_vector,
    char_vector
])


print("\nFeature vector shape:")
print(message_vector.shape)


# ==========================================
# 5. BEST MODEL PREDICTION
# ==========================================

best_prediction = best_model.predict(message_vector)[0]

best_score = best_model.decision_function(message_vector)[0]

print("\n==========================================")
print("BEST MODEL (Linear SVM)")
print("==========================================")

print(f"Decision score: {best_score:.4f}")

if best_prediction == 1:
    print("Result: SPAM")
else:
    print("Result: HAM")


# ==========================================
# 6. LOGISTIC REGRESSION
# ==========================================

lr_prediction = lr_model.predict(message_vector)[0]
lr_probability = lr_model.predict_proba(message_vector)[0]

print("\n==========================================")
print("LOGISTIC REGRESSION")
print("==========================================")

print(f"Ham  : {lr_probability[0] * 100:.2f}%")
print(f"Spam : {lr_probability[1] * 100:.2f}%")

if lr_prediction == 1:
    print("Result: SPAM")
else:
    print("Result: HAM")


# ==========================================
# 7. LINEAR SVM
# ==========================================

svm_prediction = svm_model.predict(message_vector)[0]
svm_score = svm_model.decision_function(message_vector)[0]

print("\n==========================================")
print("LINEAR SVM")
print("==========================================")

print(f"Decision score: {svm_score:.4f}")

if svm_prediction == 1:
    print("Result: SPAM")
else:
    print("Result: HAM")