import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# VADER lexicon download (Lexicon-based approach for rule-based analysis)
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def get_tfidf_representation(cleaned_tokens):

    
    if not cleaned_tokens:
        return pd.DataFrame()
        
    text_corpus = [" ".join(cleaned_tokens)]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_corpus)
    
    
    feature_names = vectorizer.get_feature_names_out()
    tfidf_values = tfidf_matrix.toarray()[0]
    
    df = pd.DataFrame([tfidf_values], columns=feature_names)
    return df

def predict_sentiment(text):
    
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = "POSITIVE"
        confidence = compound  # normalized value approx
    elif compound <= -0.05:
        sentiment = "NEGATIVE"
        confidence = abs(compound)
    else:
        sentiment = "NEUTRAL"
        confidence = 1.0 - abs(compound)
        
    # Scale confidence properly between 0.5 and 0.99 for aesthetic UI if needed
    confidence = max(0.5, min(confidence, 0.99))
        
    return sentiment, round(confidence, 2)

