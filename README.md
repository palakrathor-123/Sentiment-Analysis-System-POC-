# Sentiment Analysis System (POC)

This is a Proof of Concept (POC) application developed using Python, Natural Language Processing (NLP), and Streamlit. The application analyzes user-provided text (reviews, comments, or feedback) and classifies its sentiment into **Positive**, **Negative**, or **Neutral** while demonstrating the underlying text preprocessing steps and numerical feature extraction live on a dashboard.

---

## 🛠️ Features & Pipeline

The application visualizes the complete NLP pipeline in real-time as requested by the technical specifications:
1. **Input Section**: Accepts sentences, paragraphs, or reviews from the user.
2. **Text Preprocessing Blocks**:
   * Lowercasing
   * Punctuation & Special Characters Removal
   * Word Tokenization
   * Stopword Removal (using NLTK libraries)
3. **Feature Extraction**: Converts processed words into numerical representations using a **TF-IDF Vectorizer** matrix display.
4. **Sentiment Prediction**: Calculates and displays the final polarity class along with a dynamic **Confidence Score** progress bar.
5. **All Results Box**: Displays a clean, aligned final log of all parameters.

---

## 📂 Project Structure

```text
sentiment-analysis-poc/
│
├── .gitignore                   # Ignores unwanted cache/local environment files
├── requirements.txt             # Python project dependencies
├── README.md                    # Project documentation and setup guide
├── app.py                       # Main Streamlit Dashboard Application (UI Logic)
│
└── src/                         # Core implementation source code
    ├── __init__.py              # Marks directory as a Python package
    ├── preprocessing.py         # Step-by-step text cleaning and tokenization logic
    └── model.py                 # Sentiment categorization and TF-IDF matrix generation
```

🚀 Setup & Installation Instructions
Follow these steps to set up and run the project locally on your system:

## 1. Clone or Open the Project
Open your project folder (sentiment-analysis-poc) in your preferred code editor (e.g., VS Code).

## 2. Install Dependencies
Open your terminal inside the project directory and install the required external libraries using pip:    
### pip install -r requirements.txt

## 3. Run the Dashboard Application
Launch the Streamlit web server by running the following command in your terminal:
### streamlit run app.py

🧪 Sample Test Inputs
To evaluate the system's performance across different polarity ranges, you can test with the following inputs:
Positive: "The customer support was amazing, and they resolved my issue within minutes! Highly recommended."
Negative: "Worst experience ever! The product arrived broken, and nobody is responding to my emails."
Neutral: "The product is okay. It performs the basic tasks but lacks advanced features."

## Technologies Used
Language: Python 3.11.x
Frontend/UI: Streamlit Framework
NLP Processing: NLTK (Natural Language Toolkit)
Feature Extraction: Scikit-learn (TF-IDF Vectorizer)
Data Handling: Pandas DataFrames

Dashboard Overview
<img width="1352" height="568" alt="Image" src="https://github.com/user-attachments/assets/dc24b40d-648a-406b-9160-bd6f9a7b66e0" />

<img width="634" height="271" alt="Image" src="https://github.com/user-attachments/assets/ddf61a84-8991-4efe-9722-963ca847fb1f" />


Author
Palak Rathore
