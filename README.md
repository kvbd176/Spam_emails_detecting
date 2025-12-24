# Spam Email Detection using LSTM (Beginner AI/ML Project)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)
![Keras](https://img.shields.io/badge/Keras-2.16-red)
![NLP](https://img.shields.io/badge/NLP-WordCloud-green)

---

## Overview

This project implements a **Spam Email Detection system** using **Python** and **deep learning**. It classifies emails as **spam** or **not spam (ham)** using Natural Language Processing (NLP) and an **LSTM (Long Short-Term Memory)** neural network.

This project was created as part of my **self-learning journey in AI/ML**, where I explored text preprocessing, neural networks, and model evaluation.

---

## Objective

- Learn and implement **NLP techniques** to process email text.  
- Build a **machine learning model** capable of classifying emails as spam or ham.  
- Gain hands-on experience with **TensorFlow and Keras**.  
- Apply foundational AI/ML concepts like **supervised learning** and **neural networks** in a practical project.

---

## Skills Demonstrated

- **Programming Languages:** Python  
- **Libraries/Frameworks:** TensorFlow, Keras, NLTK, Pandas, NumPy, Matplotlib, Seaborn, WordCloud  
- **AI/ML Concepts:** Supervised learning, neural networks, LSTM, text preprocessing, tokenization, sequence padding  
- **Data Analysis:** Exploratory Data Analysis (EDA), word cloud visualization  
- **Problem Solving:** Handling imbalanced dataset, cleaning text data, building an end-to-end ML pipeline  

This project shows **my ability to learn new technologies quickly** and implement them effectively.

---

## Dataset

- Contains **5171 emails** with labels indicating spam (1) or ham (0).  

| Column      | Description |
|------------|-------------|
| `Unnamed: 0` | Index (not used) |
| `label`      | "ham" or "spam" |
| `text`       | Email content |
| `label_num`  | 0 = ham, 1 = spam |

**Challenge:** Imbalanced dataset (more ham than spam).  
**Solution:** Downsampled ham emails to match the number of spam emails.

---

## Project Steps

### 1. Data Preprocessing

- Removed “Subject” from emails.  
- Removed **punctuations** and **stopwords**.  
- Converted text to lowercase and tokenized words.

```python
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def remove_stopwords(text):
    return " ".join([word.lower() for word in text.split() if word.lower() not in stop_words])
```
### 2. Exploratory Data Analysis (EDA)

- Checked spam vs ham distribution.
- Visualized most frequent words using WordClouds.

```python
from wordcloud import WordCloud
wc = WordCloud(background_color='black', max_words=100).generate(email_text)
```
### 3. Text to Numerical Vectors

- Tokenized emails into sequences of numbers.
- Padded sequences to equal length for neural network input.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)
train_sequences = tokenizer.texts_to_sequences(train_X)
train_sequences = pad_sequences(train_sequences, maxlen=100, padding='post')
```

### 4. Model Building

- Built an LSTM neural network:
    - Embedding → LSTM → Dense → Dropout → Output layer
- Binary classification using sigmoid activation.

```python
import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### 5. Model Training

- Split data into train (80%) and test (20%).
- Used EarlyStopping and ReduceLROnPlateau callbacks.
- Achieved ~96% test accuracy.

```python
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5)

history = model.fit(train_sequences, train_Y, validation_data=(test_sequences, test_Y),
                    epochs=20, batch_size=32, callbacks=[es, lr])
```
### 6. Model Evaluation

- Training vs Validation Accuracy
  ```python
  import matplotlib.pyplot as plt
  plt.plot(history.history['accuracy'], label='Training Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend()
  plt.show()
  ```
- Test Accuracy: 96.8%
- Confirms the model can effectively distinguish spam from ham.

### Key Learnings

- Hands-on experience with AI/ML libraries: TensorFlow, Keras, NLTK.
- Learned text data preprocessing and sequence modeling.
- Understood LSTM neural networks for NLP tasks.
- Developed problem-solving skills for dataset balancing and model tuning.
- Gained experience in model evaluation and visualization.

### Future Improvements

- Experiment with bigger or bidirectional LSTM models to capture more information from text.
- Build a simple interface to test your model using HTML, CSS, and JavaScript, connecting it to a PHP backend or MySQL database for storing results.
- Work on larger datasets to make the model more robust.

### Tools and Libraries

- Python,Pandas,NumPy
- NLTK,WordCloud
- Matplotlib,Seaborn
- TensorFlow,Keras
