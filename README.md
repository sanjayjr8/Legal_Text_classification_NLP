# Legal Text Classification with Machine Learning and NLP

## 🔹 Project Overview

This project aims to **automate the process of legal text classification** by applying **Machine Learning (ML) and Natural Language Processing (NLP)** techniques.  
We have implemented **Random Forest, LSTM, and Stacked Ensemble models** to accurately categorize legal documents into their respective classes.

---

## 🔹 The Problem

In the legal domain, vast amounts of documents — judgments, legislation, and case files — are generated daily.  
Organizing and retrieving these documents efficiently is a major challenge for lawyers, judges, and legal firms.  
Manual review is not only time-intensive but prone to human error.  
This project addresses this bottleneck by designing a pipeline to **automate the process of legal text classification** with high accuracy.

---

## 🔹 Our Approach (Technical)

### 1️⃣ **Data Collection and Inspection**

- We used a custom legal text dataset with **7 classes**:
  - CITED
  - APPLIED
  - REFERRED
  - FOLLOWED
  - RELATED
  - DISCUSSED
  - [Class 7 Name]
  
- The dataset was **imbalanced**, with some classes underrepresented.

---

### 2️⃣ **Data Preprocessing (NLP Techniques)**

To maximize the classifier’s ability to learn from raw text, we performed extensive text cleansing and transformation:

✅ **Text Tokenization**  
✅ **Removal of Stop Words** (with NLTK’s standard list)  
✅ **Lemmatization** (with WordNet Lemmatizer) — reducing words to their base forms (running → run)  
✅ **Lowercasing** — for uniform representation  
✅ **Handling of Class Imbalance** (with SMOTE or ADASYN) — to produce a more balanced training set.  

Such careful text processing helps to reduce noise, standardize inputs, and enable our models to learn more robust patterns.

---

### 4️⃣ **Model Implementations**

We implemented **3 different models**:

**➥ Random Forest (with TF-IDF vectorization)**  
Provides strong baseline performance with interpretability.

**➥ LSTM (with Embedding Layers)**  
Leverages semantic relationships in text — especially helpful for context-dependent phrases in legal documents.

**➥ Stacked Ensemble**  
Combination of both the above to produce more robust and accurate predictions.

---

### 5️⃣ **Evaluation Metrics**

To account for class imbalance and to gauge true performance across all classes, we used:

✅ Accuracy   
✅ Precision   
✅ Recall   
✅ F1-Score (with per-class metrics)   
✅ Confusion Matrix (with per-label breakdown)  

This lets us identify which classes perform well and where we need further improvement.

---

## 🔹 How We Handled Class Imbalance

Class imbalance can undermine classifier performance — causing it to be biased toward majority classes.  
To address this:

✅ We used **SMOTE (Synthetic Minority Oversampling)** to generate additional samples for underrepresented classes.  
✅ This resulted in a much more **balanced training set**, which directly improved classifier performance across all classes.  
✅ The F1 score for previously underrepresented classes improved by up to 20%.  

---

## 🔹 Improvement in Accuracy

Originally, with imbalanced classes, we were seeing an accuracy of roughly **75% with poor F1 scores for many classes**.

After applying SMOTE and fine-tuning hyperparameters (with grid search) for both the LSTM and the Random Forest, and then employing **stacked ensembling**, we:

✅ Increased the accuracy to **90%+**   
✅ F1 score for each class surpassed **0.85**   
✅ Overall robustness and generalization improved — reducing overfitting.

---

## 🔹 Summary of NLP Techniques Utilized

✅ Text Tokenization
✅ Stop Words Removal
✅ Lemmatization
✅ Text Normalization
✅ SMOTE Oversampling
✅ TF-IDF Vectorization
✅ Embedding Layers with LSTM
✅ Stacked Model for Better Performance
✅ Detailed Evaluation with F1, Accuracy, and Classification Report per class

---

## 🔹 Tools and Libraries

- **Python 3.x**
- **Scikit-learn**
- **TensorFlow/Keras** (for LSTM)
- **NLTK** (for text processing)
- **Pandas**, **NumPy**, and **Matplotlib** (for data manipulations)

---

## 🔹 Conclusion

This pipeline successfully demonstrates **the power of Machine Learning and NLP** in **automating legal document classification** — even under challenging conditions stemming from class imbalance.  
Through careful data preparation, algorithm selection, and evaluation, we have significantly improved both the accuracy and robustness of the classifier.

---


✅ Experiment with **Transformer models (BERT, RoBERTa)** for even greater accuracy.  
✅ Implement **active learning** to aid human reviewers in improving the training set.  
✅ Integrate a **REST API** for real-world application.  
✅ Develop a **web application with Streamlit** for easy and interactive usage.


