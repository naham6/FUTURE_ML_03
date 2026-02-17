# AI-Powered Resume Screening System

An automated **Machine Learning Ranking System** that reads resumes, parses job descriptions and ranks candidates based on relevance. It uses **NLP (TF-IDF, Lemmatization)** to extract skills and **KNN (K-Nearest Neighbors)** to score candidates against a job description.

---

## 🚀 Overview

Hiring teams often receive hundreds of resumes for a single role. Manually screening them is slow and biased.

This project builds a **Resume Screening Engine** that:

1. **Cleans & Normalizes** text (removing stop words, lemmatizing).
2. **Vectorizes** resumes into a mathematical format (TF-IDF).
3. **Classifies** resumes into categories (e.g., HR, Engineering, Sales) using **SVM**.
4. **Ranks** candidates by similarity to a Job Description using **KNN**.
5. **Visualizes** the "Skill Gap" (what the candidate is missing).

---

## 📊 Key Features

- **Smart Text Cleaning:** Uses **NLTK Lemmatization** to understand context (e.g., treats "managing" and "manager" as the same skill).
- **Automated Categorization:** Classifies resumes into 24 distinct industry categories with **67% Accuracy** (outperforming Random Forest).
- **Ranked Search:** Returns a list of top candidates sorted by a **Match Confidence** score (0–100%).
- **Skill Gap Analysis:** Visually compares a candidate's skills vs. the job description to highlight missing keywords.
- **Visualization:** Includes Confusion Matrices, Category Distributions and Word Clouds.

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Machine Learning:** Scikit-Learn (SVM, KNN, Random Forest, XGboost)  
- **NLP:** NLTK (Tokenization, Lemmatization), spaCy (Entity Extraction), TfidfVectorizer  
- **Data Manipulation:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn, WordCloud  

---

## ⚙️ How It Works (The Pipeline)

### 1️⃣ Preprocessing

- Regex cleaning (removing URLs, special characters)
- NLTK Tokenization & Stop-word removal
- **Lemmatization:** Converting words to their root form (e.g., "Running" → "Run")

### 2️⃣ Feature Extraction

- **TF-IDF (Term Frequency–Inverse Document Frequency):**  
  Converts text into numerical vectors, giving more weight to unique/important words.

### 3️⃣ Modeling

- **Classification:** A **Linear Support Vector Machine (SVM)** predicts the job category.
- **Ranking:** A **K-Nearest Neighbors (KNN)** algorithm calculates the Cosine Distance between the Job Description vector and Resume vectors.

---

## 📉 Performance Results

We compared multiple models for the classification task:

| Model | Accuracy | Notes |
|-------|----------|-------|
| **Linear SVM** | **67.40%** | Best performer on high-dimensional text data |
| Random Forest | 61.97% | Struggled with sparse matrices |
| Naive Bayes | 53.11% | Good baseline, but less accurate than SVM |

**Search Engine Performance:**  
The KNN model successfully retrieves relevant resumes (e.g., searching for "HR Manager" returns 100% HR-category resumes).

---

## 📸 Visualization

The system generates visual reports for recruiters:

### 1️⃣ Skill Gap Analysis
A visual breakdown of which skills the candidate matches and which are missing from the job description.

### 2️⃣ Confusion Matrix
Shows exactly where the model confuses similar categories (e.g., "Sales" vs. "Business Development").

### 3️⃣ Word Clouds
Generates word clouds to show the most dominant skills in any specific industry (e.g., "Python", "SQL" for IT).

---

## 💻 Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/nahame/FUTURE_ML_03.git
cd resume-screening-system
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3️⃣ Run the System

```bash
python main.py
```

### 4️⃣ Interactive Mode

The script will prompt you to enter a Job Description or Resume snippet.
It will then output the:

* **Predicted Category**
* **Match Confidence**

---

## 🔮 Future Improvements

* **Deep Learning:** Implement BERT/Transformers for better context understanding
* **Resume Parsing:** Integrate a PDF parser (like `pdfplumber`) to read actual files instead of text snippets

---

Contributions are welcome!
Feel free to fork the repository and submit a pull request.

---
