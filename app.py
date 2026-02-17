import streamlit as st
import pickle
import re
import nltk

# --- 1. Setup & Config ---
st.set_page_config(page_title="AI Resume Screener", page_icon="VX", layout="wide")

# Download NLTK data quietly
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 2. Load Models (Cached) ---
@st.cache_resource
def load_models():
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('knn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('svm_classifier.pkl', 'rb') as f:
            clf = pickle.load(f)
        return tfidf, model, clf
    except FileNotFoundError:
        st.error("Error: Model files not found. Please upload .pkl files.")
        return None, None, None

tfidf, model, clf = load_models()

# --- 3. Cleaning Function ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_resume(text):
    # Basic cleanup
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # NLTK processing
    words = nltk.word_tokenize(text)
    filtered_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

# --- 4. UI Layout ---
st.title("📄 AI-Powered Resume Screener")
st.markdown("### Intelligent Resume Analysis & Ranking System")

# Create Tabs
tab1, tab2 = st.tabs(["🔍 Resume Scanner", "📊 Skill Gap Analyzer"])

# --- TAB 1: General Scanner ---
with tab1:
    st.subheader("Predict Job Category")
    resume_text = st.text_area("Paste Resume Text Here:", height=200, key="resume_input")
    
    if st.button("Analyze Resume"):
        if resume_text and tfidf:
            # 1. Clean
            cleaned_text = clean_resume(resume_text)
            
            # 2. Vectorize
            vector = tfidf.transform([cleaned_text])
            
            # 3. Predict Category
            category = clf.predict(vector)[0]
            
            # 4. Find Match Confidence
            distances, indices = model.kneighbors(vector)
            match_score = (1 - distances[0][0]) * 100
            
            # Display Results
            st.success(f"**Predicted Category:** {category}")
            st.info(f"**Relevance Score:** {match_score:.2f}%")
            
        elif not resume_text:
            st.warning("Please paste a resume first.")

# --- TAB 2: Skill Gap Analysis (The Notebook Logic) ---
with tab2:
    st.subheader("Compare Resume vs. Job Description")
    
    col1, col2 = st.columns(2)
    with col1:
        jd_text = st.text_area("Paste Job Description:", height=150)
    with col2:
        res_text_gap = st.text_area("Paste Resume (for Gap Analysis):", height=150)
        
    if st.button("Check Missing Skills"):
        if jd_text and res_text_gap:
            # Clean both
            resume_words = set(clean_resume(res_text_gap).split())
            jd_words = set(clean_resume(jd_text).split())
            
            # Logic
            required_skills = {word for word in jd_words if len(word) > 2} # Filter short noise
            missing_skills = required_skills - resume_words
            match_count = len(required_skills) - len(missing_skills)
            
            # Display
            if required_skills:
                score = (match_count / len(required_skills)) * 100
            else:
                score = 0
            
            st.metric("Skill Match Score", f"{score:.1f}%")
            
            if missing_skills:
                st.error(f"⚠️ **Missing Skills:** {', '.join(list(missing_skills)[:15])}")
            else:
                st.success("✅ No major skills missing!")
        else:
            st.warning("Please fill in both text boxes.")
