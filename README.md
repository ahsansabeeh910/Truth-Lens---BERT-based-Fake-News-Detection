# TruthLens — Fake News Detection using BERT



TruthLens modern fake news detection system built using **BERT Transformers**, **PyTorch**, and **Streamlit**. The application analyzes news articles and predicts whether the news is **Real** or **Fake** with confidence scores.

---

# Features

- Fake vs Real News Classification
- Fine-tuned BERT Transformer Model
- Modern Streamlit Frontend
- Confidence Score Prediction
- Real-time Text Analysis
- NLP + Deep Learning Based Detection

---

# Tech Stack

- Python
- Streamlit
- HuggingFace Transformers
- BERT (`bert-base-uncased`)
- PyTorch
- Pandas
- Scikit-learn

---

# Project Structure

```bash
Fake-News-Detection/
│
├── app.py
├── requirements.txt
├── Fake.csv
├── True.csv
│
└── fake_news_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/ahsansabeeh910/fake-news-detection.git
cd fake-news-detection
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Application

```bash
streamlit run app.py
```

---

# Model Training

The model is trained using:

- Fake.csv
- True.csv

Dataset preprocessing includes:
- Text cleaning
- Label encoding
- Tokenization using BERT tokenizer

---

# Example Inputs

## Real News

```text
NASA confirms discovery of water traces on Mars during latest rover mission.
```

## Fake News

```text
Aliens officially elected as presidents by secret world government.
```

---

# Future Improvements

- Live News API Integration
- Explainable AI Predictions
- Multilingual Support
- Browser Extension
- News Source Credibility Analysis

---

# Author

Developed by **Sabeeh Ahsan**

Jaypee Institute of Information Technology

---

# ⭐ If you like this project

Give it a ⭐ on GitHub!
