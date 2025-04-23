import nltk
import pandas as pd
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

# Mengabaikan peringatan
warnings.filterwarnings("ignore")

# Mengunduh data yang diperlukan
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset penyakit dan pengobatan
disease_data = pd.read_excel('Diseases.xlsx')
treatment_data = pd.read_excel('Pengobatan.xlsx')

# Pastikan tidak ada data kosong
disease_data['Symptoms'] = disease_data['Symptoms'].fillna('No symptoms listed')

# Gabungkan gejala menjadi satu string besar
raw = disease_data['Symptoms'].str.cat(sep=" ").lower()

# Tokenisasi
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# Lemmatization
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def is_question(text):
    return text.strip().endswith('?')

# Menggabungkan dataset penyakit dan pengobatan berdasarkan label penyakit
merged_data = pd.merge(disease_data, treatment_data, on="Condition", how="left")

def disease_prediction(user_response, top_n=3, threshold=0.2):
    if not is_question(user_response):
        return "Harap ajukan pertanyaan agar saya bisa membantu."
    
    symptoms_list = disease_data['Symptoms'].tolist()
    symptoms_list.append(user_response)
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(symptoms_list)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Ambil penyakit yang memiliki similarity di atas threshold
    valid_indices = [idx for idx in np.argsort(cosine_sim)[-top_n:][::-1] if cosine_sim[idx] > threshold]

    if len(valid_indices) > 0:
        total_score = sum(cosine_sim[valid_indices])  # Normalisasi skor untuk persentase
        
        result = "\n"
        
        for i, idx in enumerate(valid_indices, start=1):
            disease_name = disease_data['Condition'].iloc[idx]
            probability = (cosine_sim[idx] / total_score) * 100  # Hitung probabilitas
            
            treatment_info = merged_data.loc[merged_data['Condition'] == disease_name, 'Treatment Options'].values
            treatment_text = treatment_info[0] if len(treatment_info) > 0 else "Tidak ada informasi perawatan yang tersedia."

            # Format daftar penyakit
            result += f"{i}. {disease_name} ({probability:.2f}%)\n\n"
            result += f"   <br><br>Cara pengobatan {disease_name.lower()}:\n <br><br>  {treatment_text}\n\n"

        return result.strip()
    
    else:
        return "Aku minta maaf, tapi aku tidak mengerti gejalanya atau tidak ada kecocokan yang cukup. Jika Anda khawatir tentang kesehatan Anda, silakan berkonsultasi dengan dokter."


# Greeting
GREETING_INPUTS = ("halo", "hi", "selamat pagi", "hey", "hai")
GREETING_RESPONSES = ["Hai, bagaimana saya bisa membantu?"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def chat(user_response):
    user_response = user_response.lower()
    
    if user_response in ["thanks", "thank you"]:
        return "Sama-sama!"
    elif greeting(user_response):
        return greeting(user_response)
    elif is_question(user_response):
        return disease_prediction(user_response)
    else:
        return "Saya hanya dapat memahami pertanyaan. Harap ajukan pertanyaan agar saya bisa membantu."