# 🛠 Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

from transformers import DistilBertTokenizer, TFDistilBertModel


# 🗂 Load Dataset
file_path = 'Kanker Data.xlsx'
dataset = pd.read_excel(file_path, sheet_name='alldata_1_for_kaggle')

# 🔄 Rename columns for easier reference
dataset.columns = ['ID', 'Label', 'Text']

# ❌ Drop rows with missing values
dataset = dataset.dropna()

# 🏷 Encode label data if not numeric
if dataset['Label'].dtype == object:
    label_encoder = LabelEncoder()
    dataset['Label'] = label_encoder.fit_transform(dataset['Label'])

# Ensure label data is numeric and binary (strictly 0 and 1)
dataset['Label'] = (dataset['Label'] > 0).astype(int)

# 🔀 Split dataset into train, validation, and test sets
X = dataset['Text']
y = dataset['Label']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# 🔗 Load DistilBERT model and tokenizer
bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# 🔌 Function to tokenize and encode text
def tokenize_and_encode(texts, tokenizer, max_length=64):
    return tokenizer(
        texts.tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="tf",
    )


# 🚀 Function to extract BERT embeddings in batches
def extract_bert_embeddings_in_batches(encoded_inputs, model, batch_size=16):
    all_embeddings = []
    for i in range(0, len(encoded_inputs['input_ids']), batch_size):
        batch_input_ids = encoded_inputs['input_ids'][i:i + batch_size]
        batch_attention_mask = encoded_inputs['attention_mask'][i:i + batch_size]

        embeddings = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
        ).last_hidden_state[:, 0, :].numpy()

        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)


# 📝 Tokenize and encode the datasets
encoded_train = tokenize_and_encode(X_train, tokenizer, max_length=64)
encoded_val = tokenize_and_encode(X_val, tokenizer, max_length=64)
encoded_test = tokenize_and_encode(X_test, tokenizer, max_length=64)

# 🧠 Extract BERT embeddings in batches
X_train_bert = extract_bert_embeddings_in_batches(encoded_train, bert_model, batch_size=16)
X_val_bert = extract_bert_embeddings_in_batches(encoded_val, bert_model, batch_size=16)
X_test_bert = extract_bert_embeddings_in_batches(encoded_test, bert_model, batch_size=16)


# 🏗 Define neural network model for BERT embeddings
model_bert = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_bert.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# 🔒 Compile model with optimized settings
learning_rate = 0.001  # Lower learning rate for stability
optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)  # Clip gradients
model_bert.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 🔄 Train the model using only BERT features
epochs = 50  # You can adjust epochs as needed
batch_size = 16
history_bert = model_bert.fit(
    X_train_bert, y_train,
    validation_data=(X_val_bert, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# 📊 Visualize training and validation accuracy and loss
plt.figure(figsize=(12, 5))

# Training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history_bert.history['accuracy'], label='Training Accuracy')
plt.plot(history_bert.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history_bert.history['loss'], label='Training Loss')
plt.plot(history_bert.history['val_loss'], label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# 🧾 Evaluate model
y_pred_bert = (model_bert.predict(X_test_bert) > 0.5).astype("int32")
print("Classification Report:\n", classification_report(y_test, y_pred_bert))