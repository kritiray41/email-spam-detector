
# 📧 End-to-End AI Email Spam Detector

An end-to-end Machine Learning web application that classifies email and SMS messages into **Ham (Legitimate)** or **Spam** using Natural Language Processing (NLP), Scikit-Learn, and Streamlit.

---

## 📌 Project Overview
Unsolicited commercial emails pose significant security risks, including phishing and malware. This project implements a full ML pipeline—from text preprocessing and TF-IDF vectorization to model evaluation, web deployment, and containerization.

Given the class imbalance in spam datasets (~87% Ham vs ~13% Spam), the model prioritizes **Precision** to ensure legitimate emails are never falsely flagged as spam.

---

## 🛠️ Tech Stack & Tools
- **Language:** Python 3.10+
- **NLP & Preprocessing:** NLTK, Regular Expressions (Regex)
- **Machine Learning:** Scikit-Learn (Multinomial Naive Bayes, Logistic Regression, Support Vector Machines)
- **Web Interface:** Streamlit
- **Serialization & Utility:** Joblib, Pandas, NumPy
- **Containerization:** Docker

---

## ⚙️ Architecture & Pipeline

```text
Raw Email Data ➡️ Text Preprocessing ➡️ TF-IDF Vectorization ➡️ ML Classifier ➡️ Streamlit App
