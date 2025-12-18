import pickle
from models.Transformers import Transformers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


sentiment_mapping = {
    0: "sadness",      
    1: "joy",          
    2: "love", 
    3: "anger", 
    4: "fear", 
    5: "surprise", 
}




# Charger la configuration
with open(r"LLMSEM/models/model_config.pkl", "rb") as f:
    config = pickle.load(f)

# Charger le tokenizer
with open(r"LLMSEM/models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Recréer le modèle avec la même architecture
model = Transformers(
    vocab_size=config['VOCAB_SIZE'],
    embed_size=config['EMBED_DIM'],
    num_heads=config['NUM_HEADS'],
    ff_dim=config['FF_DIM']
)
model.build(input_shape=(None, config['MAX_LEN']))

# Compiler le modèle
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Charger les poids
model.load_weights(r"LLMSEM/models/transformers_weights.weights.h5")

test_texts = [
    # Niveau 1 : clair et direct
    "I am very happy today!",
    "I feel so angry about this.",
    "I am terrified of the dark.",
    "I feel so sad and lonely.",
    "What a surprising turn of events!",
    "I love spending time with my family.",

    # Niveau 2 : phrases un peu nuancées
    "I can't believe how lucky I am.",
    "He disappointed me so much.",
    "Walking alone at night makes me nervous.",
    "Missing her makes me feel down.",
    "I didn't expect that to happen!",
    "I really care about my friends.",

    # Niveau 3 : expressions idiomatiques et sarcasme
    "Oh great, another rainy day… just what I needed.",
    "I am on cloud nine with this news.",
    "My heart sank when I heard the truth.",
    "She left me, and I’m not even surprised.",
    "Well, that was unexpected!",
    "Nothing beats a warm hug from a loved one.",

    # Niveau 4 : phrases ambiguës ou multiples émotions
    "I am happy but also a little nervous about tomorrow.",
    "I love her, but she makes me so frustrated.",
    "I fear the worst but hope for the best.",
    "I am sad and relieved at the same time.",
    "The surprise party made me anxious yet thrilled.",
    "I love this movie, though some parts were sad."
]

for text in test_texts:
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=config['MAX_LEN'])
    prediction = model.predict(padded)
    predicted_index = np.argmax(prediction, axis=-1)[0]
    predicted_sentiment = sentiment_mapping[predicted_index]
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {predicted_sentiment}")
    print(f"All Scores: {np.round(prediction[0], 2)}\n")


