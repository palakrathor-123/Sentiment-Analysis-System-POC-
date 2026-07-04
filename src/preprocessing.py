import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK data downloads 
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')  

def pipeline_preprocessing(text):
    
    # Step 1: Original Text
    step1_orig = text
    
    # Step 2: Lowercase
    step2_lowered = text.lower()
    
    # Step 3: Remove Punctuation & Special Characters
    step3_cleaned = re.sub(r'[^a-zA-Z\s]', '', step2_lowered)
    
    # Step 4: Tokenization
    step4_tokens = word_tokenize(step3_cleaned)
    
    # Step 5: Stopword Removal
    stop_words = set(stopwords.words('english'))
    step5_no_stopwords = [word for word in step4_tokens if word not in stop_words]
    
    # Final Cleaned Text as a String
    cleaned_text_str = " ".join(step5_no_stopwords)
    
    return {
        "original": step1_orig,
        "lowercase": step2_lowered,
        "no_punctuation": step3_cleaned,
        "tokens": step4_tokens,
        "no_stopwords": step5_no_stopwords,
        "final_string": cleaned_text_str
    }


