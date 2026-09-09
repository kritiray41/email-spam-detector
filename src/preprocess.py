import re

def clean_text(text: str) -> str:
    # Defang and mask URLs
    text = text.replace("[.]", ".").replace("(.)", ".")
    url_pattern = r'(https?://\S+|hxxps?://\S+|www\.\S+)'
    return re.sub(url_pattern, '<URL>', text, flags=re.IGNORECASE)
