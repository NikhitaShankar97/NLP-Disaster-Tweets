# 🏆 Natural Language Processing with Disaster Tweets
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?style=flat&logo=kaggle)](https://www.kaggle.com/competitions/nlp-getting-started)

## 📌 Overview
This project focuses on **classifying disaster-related tweets** using **Natural Language Processing (NLP)**. It was developed as part of the **Kaggle NLP Competition - "Natural Language Processing with Disaster Tweets"**, where the goal is to build a machine learning model that can **differentiate between real disaster tweets and non-disaster tweets**.

## 📂 Dataset
The dataset consists of labeled tweets along with metadata such as **keywords and locations**:

**Data Files:**
- `train.csv`: Contains **7,613 tweets** with labels (`1 = Disaster`, `0 = Not Disaster`).
- `test.csv`: Contains **3,263 tweets** without labels (used for predictions).
- `sample_submission.csv`: Provides the expected submission format.

### 📊 Exploratory Data Analysis (EDA)
Key insights from the **EDA**:
- **Keyword Analysis:** Some keywords (e.g., `"wildfire"`, `"earthquake"`, `"flood"`) strongly indicate real disasters.
- **Tweet Length:** Disaster tweets tend to be slightly longer in word count.
- **Hashtags & Mentions:** Disaster tweets frequently contain URLs (news links) and hashtags.
- **Location Data:** Around **33% of tweets lack location metadata**, reducing its reliability as a feature.

## 🔬 Approach
This project explores **multiple NLP techniques**:

1. **Traditional ML Models**:
   - **TF-IDF + Logistic Regression**
   - **Naive Bayes Classifier**
   
2. **Deep Learning Approaches**:
   - **LSTMs, GRUs**
   
3. **Transformer-based Models**:
   - **BERT** (`bert-base-uncased`)
   - **DistilBERT** (`distilbert-base-uncased`)
   - **Bertweet** (`vinai/bertweet-base`)

## 🏢 Model Pipeline
1. **Text Preprocessing**:
   - Tokenization using **Hugging Face Transformers**
   - Removing URLs, mentions, hashtags, and stopwords
   - Handling missing values for **keywords & locations**

2. **Feature Engineering**:
   - Word and character count features
   - Special token frequencies (URLs, hashtags, mentions, exclamations)
   - Embeddings from **pre-trained transformers**

3. **Training & Validation Strategy**:
   - **3-Fold Stratified Cross-Validation**
   - **Early Stopping** to prevent overfitting
   - **Gradient Accumulation** for better memory efficiency

4. **Optimization Techniques**:
   - Learning Rate Scheduling (**Cosine Decay with Warmup**)
   - Mixed Precision Training (**FP16** for speedup)
   - Hyperparameter tuning using **Optuna/W&B Sweeps**

## 📊 Results
| **Model**        | **Validation Accuracy** | **F1 Score** | **Inference Time** |
|----------------|-------------------|------------|----------------|
| TF-IDF + Logistic Regression | 0.78 | 0.74 | ⚡ Very Fast |
| DistilBERT | 0.83 | 0.81 | 🚀 Fast |
| BERTweet (Final Model) | **0.86** | **0.84** | ⚡⚡ Moderate |

- The **BERTweet model performed the best**, thanks to its training on social media text.
- **DistilBERT provided a great balance of speed and accuracy**.
- **Traditional models (TF-IDF + Logistic Regression) lacked contextual understanding** but were computationally efficient.

## 🚀 Running the Project
### **1️⃣ Setup Environment**
First, install the required dependencies:
```bash
pip install -r requirements.txt
```

### **2️⃣ Run Preprocessing & Training**
Execute the notebook **`nlp_disaster_tweets_classification.ipynb`** to preprocess data and train models.

```bash
jupyter notebook nlp_disaster_tweets_classification.ipynb
```

## 👨‍💻 Authors
- **Nikhita Shankar**  
- **Shatakshi Bhatnagar** 

## 💪 Acknowledgments
- **Kaggle for hosting the competition**
- **Hugging Face for transformer models**
- **Matplotlib, Seaborn, and Pandas for visualizations**
- **Scikit-learn for ML models**
