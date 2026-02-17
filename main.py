import pickle
import re
import nltk

# Ensure NLTK data is downloaded (runs once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading necessary NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1. Load the Saved Models ---
print("Loading system models...")
try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('knn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('svm_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)
    print("Models loaded successfully!")
except FileNotFoundError:
    print("Error: Model files not found. Make sure .pkl files are in the same folder.")
    exit()

# --- 2. The NLTK Cleaning Function ---
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
   
    # NLTK Tokenization & Lemmatization
    words = nltk.word_tokenize(text)
    filtered_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

# --- 3. The Core System ---
def screen_resume(resume_text):
    # A. Clean
    cleaned_text = clean_resume(resume_text)
   
    # B. Vectorize
    vector = tfidf.transform([cleaned_text])
   
    # C. Predict Category
    category = clf.predict(vector)[0]
   
    # D. Search for Similar Resumes (Simulated)
    # This finds the "Nearest Neighbors" in the trained space
    distances, indices = model.kneighbors(vector)
    match_score = (1 - distances[0][0]) * 100
   
    return {
        "Predicted Category": category,
        "Match Confidence": f"{match_score:.2f}%"
    }

# --- 4. Interactive Test ---
if __name__ == "__main__":
    print("\n--- RESUME SCREENING SYSTEM ---")
    print("Enter a resume snippet (or type 'exit'):")
   
    while True:
        user_input = input("\nResume Text: ")
        if user_input.lower() == 'exit':
            break
           
        if len(user_input) < 10:
            print("Please enter a longer text.")
            continue
           
        result = screen_resume(user_input)
       
        print("\n--- ANALYSIS REPORT ---")
        print(f"📂 Predicted Category: {result['Predicted Category']}")
        print(f"📊 Relevance Score:    {result['Match Confidence']}")
        print("-----------------------")
