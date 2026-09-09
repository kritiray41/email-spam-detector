# AI Email Spam & Phishing Detector

An offline-capable, context-aware machine learning application designed to detect modern spear-phishing and Business Email Compromise (BEC). This project has been upgraded from a basic TF-IDF/SVM baseline to a sophisticated hybrid neural network pipeline.

## Model Architecture

This application utilizes a **Hybrid ML Pipeline** to balance deep semantic understanding with extreme tabular efficiency:
* **Feature Extraction:** A frozen `distilbert-base-uncased` transformer processes the raw email text and outputs a 768-dimensional semantic embedding array.
* **Classification:** An XGBoost (Gradient Boosting) classifier evaluates the dense numerical representations to draw hard, non-linear decision boundaries, preventing majority features ("corporate speak") from washing out high-signal minority features (urgency, masked links).
* **Preprocessing:** Custom URL defanging (e.g., converting `[.]` to `.`) and masking (tokenizing as `<URL>`) is applied prior to inference to prevent feature collapse.

##  Repository Structure

* `app.py`: The main Streamlit web application.
* `src/predict.py`: Handles model loading, embedding extraction, and XGBoost inference.
* `src/preprocess.py`: Contains regex logic for URL masking and text defanging.
* `models/`: Directory for the downloaded Hugging Face and XGBoost artifacts (Ignored by Git due to size limits).
* `requirements.txt`: Project dependencies.

*(Note: Model training and dataset blending are handled externally via Google Colab to leverage GPU acceleration.)*

## Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/kritiray41/email-spam-detector.git](https://github.com/kritiray41/email-spam-detector.git)
cd email-spam-detector
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download Offline Model Artifacts**
Because the `distilbert-base-uncased` safetensors exceed GitHub's standard file limits, the model weights are not included in this repository.

* Extract your trained model zip file.
* Place the contents (`config.json`, `model.safetensors`, `tokenizer.json`, `xgboost_spam.json`, etc.) directly into `models/offline_hybrid_model/`.

**5. Run the Application**
```bash
streamlit run app.py
```
## 👤 Author
**Kriti Ray**

