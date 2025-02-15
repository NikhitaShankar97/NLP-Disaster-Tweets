# 🏆 Natural Language Processing with Disaster Tweets  
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?style=flat&logo=kaggle)](https://www.kaggle.com/competitions/nlp-getting-started)  

## 📌 Overview  
This project is part of the **Kaggle NLP Competition - Natural Language Processing with Disaster Tweets**. The objective is to build a **machine learning model that classifies tweets as either related to real disasters or not**.  

Tweets often contain **metaphors, sarcasm, or unrelated context**, making this an **interesting and challenging NLP task** that goes beyond simple keyword matching.  

---

## 💂 Dataset  
The dataset consists of tweets labeled as **disaster (1) or non-disaster (0)**, along with metadata such as **keywords and location**.  

### 🌟 Data Files:
- **`train.csv`** → 7,613 tweets with labels (`target = 1` for disaster, `target = 0` for non-disaster)  
- **`test.csv`** → 3,263 tweets without labels (used for predictions)  
- **`sample_submission.csv`** → Example submission format for Kaggle  

---

## 🔍 Exploratory Data Analysis (EDA)  
Key observations from **EDA & feature analysis**:  
👉 **Keyword Influence**: Some words (e.g., `"earthquake"`, `"wildfire"`, `"flood"`) are strong indicators of disaster tweets.  
👉 **Tweet Length**: Disaster tweets **tend to be slightly longer** (word & character count).  
👉 **Special Tokens**: More **hashtags, URLs (news links), and exclamation marks** appear in disaster-related tweets.  
👉 **Location Data**: **~33% of tweets lack location info**, reducing its predictive power.  

---

## 🏰 Model Pipeline  
This project explores multiple NLP techniques, ranging from traditional machine learning models to state-of-the-art transformers.  

### **1️⃣ Text Preprocessing**
👉 Tokenization using **Hugging Face Transformers**  
👉 Lowercasing, removing URLs, mentions, hashtags  
👉 Handling missing metadata (`no_keyword`, `no_location`)  

### **2️⃣ Feature Engineering**
👉 Extracting word & character count features  
👉 Analyzing special tokens (hashtags, mentions, URLs, exclamations)  
👉 Embeddings from pre-trained transformers  

### **3️⃣ Model Training & Validation**
👉 **Baseline Models**: TF-IDF + Logistic Regression  
👉 **Deep Learning Approaches**: LSTMs, GRUs  
👉 **Transformer Models**:  
   - **BERT (`bert-base-uncased`)**  
   - **DistilBERT (`distilbert-base-uncased`)**  
   - **BERTweet (`vinai/bertweet-base`)** → 🚀 **Final Model**  

👉 **Training Strategies**
- **3-Fold Stratified Cross-Validation**  
- **Early Stopping** (to prevent overfitting)  
- **Gradient Accumulation** (for better memory efficiency)  
- **Learning Rate Scheduling** (**Cosine Decay with Warmup**)  

---

## 📊 Results & Insights  
| **Model**        | **Validation Accuracy** | **F1 Score** | **Inference Time** |
|----------------|-------------------|------------|----------------|
| TF-IDF + Logistic Regression | 0.78 | 0.74 | ⚡ Very Fast |
| DistilBERT | 0.83 | 0.81 | 🚀 Fast |
| **BERTweet (Final Model)** | **0.86** | **0.84** | ⚡⚡ Moderate |

👉 The **BERTweet model** achieved the highest accuracy and F1-score, leveraging its **pretraining on social media text**.  
👉 **DistilBERT provided a balance** between speed and accuracy.  
👉 **Traditional models performed decently** but lacked contextual understanding of tweets.  

---

## 📈 Next Steps  
👉 **Fine-tune hyperparameters further** using Optuna/W&B Sweeps  
👉 **Experiment with GPT-based models** for improved contextual understanding  
👉 **Deploy the best model** as an API for real-time disaster tweet classification  

---

## 🤝 Connect with Me  
If you found this project interesting or have any suggestions, feel free to connect!  

📧 **Email**: [nikhita.shankar97@gmail.com](mailto:nikhita.shankar97@gmail.com)  
👉 **LinkedIn**: [linkedin.com/in/nikhita-shankar-/](https://linkedin.com/in/nikhita-shankar-/)  

---

### ⭐ If you found this useful, consider giving the repository a star! ⭐  
