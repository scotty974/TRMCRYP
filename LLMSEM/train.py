import tensorflow as tf
from models.Transformers import Transformers
from utils.data_bourse import balanced_df
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

VOCAB_SIZE = 10000
EMBED_DIM = 128
NUM_HEADS = 6
FF_DIM = 512
BATCH_SIZE = 32
MAX_LEN = 200
NUM_CLASSES = 3

# Découpage train / val / test
X_train_full, X_temp, y_train_full, y_temp = train_test_split(
    balanced_df["Sentence"], balanced_df["Sentiment"], test_size=0.3, stratify=balanced_df["Sentiment"], random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
y_test_original = y_test.copy()

# Tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(X_train_full)

X_train = tokenizer.texts_to_sequences(X_train_full)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_val = pad_sequences(X_val, maxlen=MAX_LEN)
X_test = pad_sequences(X_test, maxlen=MAX_LEN)

y_train = tf.keras.utils.to_categorical(y_train_full, num_classes=NUM_CLASSES)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

# Calcul des poids de classes
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_full),
    y=y_train_full
)
class_weight = dict(zip(np.unique(y_train_full), weights))

# Modèle
model = Transformers(
    vocab_size=VOCAB_SIZE,
    embed_size=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM
)

precision_macro = tf.keras.metrics.Precision(name="precision_macro")
recall_macro = tf.keras.metrics.Recall(name="recall_macro")
f1_macro = tf.keras.metrics.F1Score(name="f1_macro")

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy", precision_macro, recall_macro, f1_macro]
)

# Entraînement avec validation
history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=10,
    validation_data=(X_val, y_val),
    class_weight=class_weight
)

# Prédictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Sauvegarde du modèle et du tokenizer
model.save_weights(r"LLMSEM/models/transformers_weights.weights.h5")

with open(r"LLMSEM/models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

config = {
    "VOCAB_SIZE": VOCAB_SIZE,
    "EMBED_DIM": EMBED_DIM,
    "NUM_HEADS": NUM_HEADS,
    "FF_DIM": FF_DIM,
    "MAX_LEN": MAX_LEN
}
with open(r"LLMSEM/models/model_config.pkl", "wb") as f:
    pickle.dump(config, f)

# Matrice de confusion
cm = confusion_matrix(y_test_original, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(NUM_CLASSES),
            yticklabels=range(NUM_CLASSES))
plt.xlabel("Classes prédites")
plt.ylabel("Vraies classes")
plt.savefig(r"LLMSEM/results/confusion_matrix.png")
plt.show()

# Accuracy entraînement
plt.figure(figsize=(10, 6))
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy entraînement")
plt.ylabel("Accuracy")
plt.xlabel("Époque")
plt.savefig(r"LLMSEM/results/accuracy.png")
plt.show()
