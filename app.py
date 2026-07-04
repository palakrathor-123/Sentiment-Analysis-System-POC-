import os
import sys
os.environ["SKLEARN_SITE_JOBLIB"] = "0"
os.environ["JOBLIB_MULTIPROCESSING_BACKEND"] = "threading"

import streamlit as st
from src.preprocessing import pipeline_preprocessing
from src.model import get_tfidf_representation, predict_sentiment

# Page setting to wide layout
st.set_page_config(page_title="Sentiment Analysis POC", layout="wide")

st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>SENTIMENT ANALYSIS – INPUT & OUTPUT</h2>", unsafe_allow_html=True)
st.markdown("---")

# Main columns create karna (Left: Input, Right: Output)
col1, col2 = st.columns(2)

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.user_input = ""

with col1:
    st.markdown("<h3 style='background-color: #1E6091; color: white; padding: 10px; text-align: center; border-radius: 5px;'>INPUT</h3>", unsafe_allow_html=True)
    
    # Text input area
    user_entry = st.text_area("Enter Text (Review/Comment):", 
                              value="I absolutely love this product! It works great and the quality is excellent.",
                              height=100)
    
    btn_click = st.button("Analyze Sentiment", type="primary")

    if btn_click or user_entry:
        st.session_state.analyzed = True
        st.session_state.user_input = user_entry

    if st.session_state.analyzed and st.session_state.user_input.strip() != "":
        # Preprocessing function hit 
        steps = pipeline_preprocessing(st.session_state.user_input)
        
        st.markdown("#### **Preprocessing Steps:**")
        
        # Step 1 Display
        st.info(f"**1️⃣ Original Text**\n{steps['original']}")
        # Step 2 Display
        st.info(f"**2️⃣ Lowercase**\n{steps['lowercase']}")
        # Step 3 Display
        st.info(f"**3️⃣ Remove Punctuation & Special Characters**\n{steps['no_punctuation']}")
        # Step 4 Display
        st.info(f"**4️⃣ Tokenization**\n{steps['tokens']}")
        # Step 5 Display
        st.info(f"**5️⃣ Stopword Removal**\n{steps['no_stopwords']}")

with col2:
    st.markdown("<h3 style='background-color: #1A936F; color: white; padding: 10px; text-align: center; border-radius: 5px;'>OUTPUT</h3>", unsafe_allow_html=True)
    
    if st.session_state.analyzed and st.session_state.user_input.strip() != "":
        steps = pipeline_preprocessing(st.session_state.user_input)
        
        # Sentiment logic and scores
        sentiment, confidence = predict_sentiment(steps['final_string'])
        tfidf_df = get_tfidf_representation(steps['no_stopwords'])
        
        # Original and Cleaned text outputs
        st.markdown(f"**Original Text:**\n<p style='color: grey;'>{steps['original']}</p>", unsafe_allow_html=True)
        st.markdown(f"**Cleaned Text:**\n<p style='font-weight: bold;'>{steps['final_string']}</p>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Feature Representation (TF-IDF Vector) Table
        st.markdown("**Feature Representation (TF-IDF Vector)**")
        if not tfidf_df.empty:
            st.dataframe(tfidf_df, hide_index=True)
        else:
            st.write("No distinct features found.")
            
        st.markdown("---")
        
        # Predicted Sentiment Display with Color logic
        st.markdown("**Predicted Sentiment:**")
        if sentiment == "POSITIVE":
            st.markdown(f"<h3 style='color: #1A936F; background-color: #E8F5E9; padding: 10px; border-radius: 5px; text-align: center;'>😊 POSITIVE</h3>", unsafe_allow_html=True)
        elif sentiment == "NEGATIVE":
            st.markdown(f"<h3 style='color: #D32F2F; background-color: #FFEBEE; padding: 10px; border-radius: 5px; text-align: center;'>😞 NEGATIVE</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color: #F57C00; background-color: #FFF3E0; padding: 10px; border-radius: 5px; text-align: center;'>😐 NEUTRAL</h3>", unsafe_allow_html=True)
            
        # Confidence score bar
        st.markdown(f"**Confidence Score:** {confidence}")
        st.progress(confidence)
        
        # All Results Summary section
        # All Results Summary section (Exact Match with Image Template)
        st.markdown("---")
        st.markdown("**All Results:**")
        
        # Ek clean gray background container box banane ke liye custom HTML
        st.markdown(f"""
        <div style="background-color: #F8F9FA; padding: 15px; border-radius: 5px; border: 1px solid #E2E8F0; color: #334155; font-size: 14px; line-height: 1.8;">
            <b style="display: inline-block; width: 140px;">• Original Text</b> : {steps['original']}<br>
            <b style="display: inline-block; width: 140px;">• Cleaned Text</b> : {steps['final_string']}<br>
            <b style="display: inline-block; width: 140px;">• Sentiment</b> : {sentiment}<br>
            <b style="display: inline-block; width: 140px;">• Confidence Score</b> : {confidence}
        </div>
        """, unsafe_allow_html=True)