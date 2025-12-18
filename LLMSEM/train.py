import tensorflow as tf
from models.Transformers import Transformers
from utils.data import train_df, validation_df, test_df
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle

VOCAB_SIZE = 10000
EMBED_DIM = 128
NUM_HEADS = 6
FF_DIM = 512
BATCH_SIZE = 32
MAX_LEN = 200

X_train = train_df["text"]
y_train = train_df["label"]
X_val = validation_df["text"]
y_val = validation_df["label"]
X_test = test_df["text"]
y_test = test_df["label"]


tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_val = pad_sequences(X_val, maxlen=MAX_LEN)
X_test = pad_sequences(X_test, maxlen=MAX_LEN)

y_test_original = y_test.copy()

y_train = tf.keras.utils.to_categorical(y_train, num_classes=6)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=6)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=6)

model = Transformers(vocab_size=VOCAB_SIZE, embed_size=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=10, validation_data=(X_val, y_val))

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# save model 
# À la fin de votre script d'entraînement
model.save_weights("transformers_weights.weights.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Sauvegardez aussi la configuration
config = {
    'VOCAB_SIZE': VOCAB_SIZE,
    'EMBED_DIM': EMBED_DIM,
    'NUM_HEADS': NUM_HEADS,
    'FF_DIM': FF_DIM,
    'MAX_LEN': MAX_LEN
}
with open("model_config.pkl", "wb") as f:
    pickle.dump(config, f)

cm = confusion_matrix(y_test_original, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(6), yticklabels=range(6))
plt.title('Matrice de Confusion')
plt.ylabel('Vraies Classes')
plt.xlabel('Classes Prédites')
plt.savefig("confusion_matrix.png")
plt.show()

# Graphique de performance
plt.figure(figsize=(10, 6))
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Précision du modèle")
plt.ylabel("Précision")
plt.xlabel("Époque")
plt.legend(["Entraînement", "Validation"], loc="upper left")
plt.savefig("accuracy.png")
plt.show()