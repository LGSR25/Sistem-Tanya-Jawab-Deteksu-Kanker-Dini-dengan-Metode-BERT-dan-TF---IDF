import nltk
import pandas as pd
import numpy as np
import random
import string
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFBertModel

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model sistem tanya jawab
qa_model = load_model('model.h5')
bert_tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
bert_model = TFBertModel.from_pretrained('indobenchmark/indobert-base-p1')

# Load dataset pengobatan
treatment_data = pd.read_excel('Pengobatan.xlsx')

# Preprocessing teks
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Hapus tanda baca
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('indonesian')]  # Hapus stopwords
    return " ".join(tokens)

# Mendapatkan keyword dari pertanyaan
def extract_keywords(question):
    processed_question = preprocess_text(question)
    return processed_question

# Prediksi penyakit menggunakan model dan pencarian pengobatan
def predict_disease_and_treatment(question):
    keywords = extract_keywords(question)
    
    # Tokenisasi dengan BERT
    inputs = bert_tokenizer(keywords, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Menggunakan representasi vektor dari CLS token
    
    # Prediksi penyakit dengan cosine similarity
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([keywords])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    disease_predictions = qa_model.predict(tfidf_matrix.toarray())  # Prediksi penyakit
    
    # Ambil penyakit dengan probabilitas tertinggi
    predicted_disease_idx = np.argmax(disease_predictions)
    predicted_disease = qa_model.classes_[predicted_disease_idx]  # Nama penyakit
    
    # Ambil informasi pengobatan berdasarkan penyakit yang diprediksi
    treatment_info = treatment_data.loc[treatment_data['Condition'] == predicted_disease, 'Treatment Options'].values
    treatment_text = treatment_info[0] if len(treatment_info) > 0 else "Tidak ada informasi perawatan yang tersedia."
    
    return f"Berdasarkan gejala yang Anda sebutkan, kemungkinan penyakit yang Anda alami adalah {predicted_disease}.\n\nCara pengobatan: {treatment_text}" 

# Sistem Chatbot
def chat(user_input):
    if user_input.lower() in ["thanks", "thank you"]:
        return "Sama-sama!"
    elif user_input.strip().endswith('?'):
        return predict_disease_and_treatment(user_input)
    else:
        return "Harap ajukan pertanyaan dengan format yang jelas."
