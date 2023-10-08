import pandas as pd
import numpy as np
import re
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from collections import Counter
from sklearn.metrics import precision_recall_curve, roc_curve

nltk.download('punkt')


# Load the dataset
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data.columns = ["label", "text"]
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Text preprocessing
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data["text"] = data["text"].apply(preprocess_text)

# Split the data
X = data["text"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization and sequence padding
max_words = 5000
max_sequence_length = 150

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Create an LSTM-based neural network
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
batch_size = 64
epochs = 20
model.fit(X_train_padded, y_train, validation_data=(X_test_padded, y_test), batch_size=batch_size, epochs=epochs, callbacks=[early_stopping])

# Make predictions
y_pred = (model.predict(X_test_padded) > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(classification_rep)

# Add Data Distribution Plot (1)
plt.figure(figsize=(10, 5))
data['label'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Spam and Ham Emails')
plt.ylabel('')
plt.show()

# Add Word Cloud Plot (2)


spam_text = ' '.join(data[data['label'] == 1]['text'])
spam_wordcloud = WordCloud(width=800, height=400).generate(spam_text)

plt.figure(figsize=(10, 5))
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Spam Emails')
plt.show()

# Add Precision-Recall and ROC Curve Plots (3)


precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Add Confusion Matrix Visualization (4)


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

history = model.fit(X_train_padded, y_train, validation_data=(X_test_padded, y_test), batch_size=batch_size, epochs=epochs, callbacks=[early_stopping])

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


def plot_top_words(texts, label, n=10):
    all_words = ' '.join(texts)
    words = all_words.split()
    word_counts = Counter(words)
    top_words = word_counts.most_common(n)

    plt.figure(figsize=(10, 5))
    plt.bar([word[0] for word in top_words], [word[1] for word in top_words])
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Top {n} Words in {label} Emails')
    plt.xticks(rotation=45)
    plt.show()


plot_top_words(data[data['label'] == 0]['text'], 'Ham', n=10)
plot_top_words(data[data['label'] == 1]['text'], 'Spam', n=10)

# Histogram of text lengths
data['text_length'] = data['text'].apply(len)
plt.figure(figsize=(10, 5))
plt.hist(data[data['label'] == 0]['text_length'], bins=50, alpha=0.5, label='Ham')
plt.hist(data[data['label'] == 1]['text_length'], bins=50, alpha=0.5, label='Spam')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.title('Distribution of Text Lengths')
plt.legend()
plt.show()
