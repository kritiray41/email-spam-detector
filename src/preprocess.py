import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords", quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    # Convert URLs into a useful token instead of deleting them
    text = re.sub(
        r"(https?://\S+|www\.\S+|hxxps?://\S+)",
        " URL ",
        text,
        flags=re.IGNORECASE
    )

    # Convert email addresses into a useful token
    text = re.sub(
        r"\b[\w.-]+@[\w.-]+\.\w+\b",
        " EMAIL ",
        text
    )

    # Keep words and useful spam-related information
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Lowercase
    text = text.lower()

    # Tokenization
    words = text.split()

    # Remove stopwords and stem
    cleaned_words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(cleaned_words)


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    sample_raw = (
        "CONGRATULATIONS! You've WON a $1,000 gift card! "
        "Claim FREE now! Visit hxxps://example.com "
        "or email test@example.com"
    )

    print("Raw:")
    print(sample_raw)

    print("\nCleaned:")
    print(clean_text(sample_raw))