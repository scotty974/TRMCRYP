import pickle
from models.Transformers import Transformers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

sentiment_mapping = {
    0: "positive",      
    1: "neutral",          
    2: "negative", 
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
    "The stock market is performing exceptionally well today!",
    "I am furious about these massive losses.",
    "I am terrified of this market crash.",
    "The portfolio decline makes me feel devastated.",
    "What a shocking market reversal!",
    "I love investing in dividend stocks.",
    
    # Niveau 2 : phrases un peu nuancées
    "I can't believe how profitable this trade was.",
    "The company earnings disappointed investors significantly.",
    "Holding this volatile stock makes me nervous.",
    "Missing that buying opportunity makes me regretful.",
    "I didn't expect the Fed to raise rates!",
    "I really trust this investment strategy.",
    
    # Niveau 3 : expressions idiomatiques et sarcasme
    "Oh great, another market correction… just what my portfolio needed.",
    "The stock price is going to the moon with this news.",
    "My heart sank when the earnings report came out.",
    "The company went bankrupt, and I'm not even surprised.",
    "Well, that merger was unexpected!",
    "Nothing beats a strong bull market rally.",
    
    # Niveau 4 : phrases ambiguës ou multiples émotions
    "I am excited but also cautious about this IPO.",
    "I love this sector, but the volatility frustrates me.",
    "I fear a recession but hope for recovery.",
    "I am disappointed yet relieved to exit that position.",
    "The surprise profit warning made me anxious yet curious.",
    "As of writing, data from TradingView showed that 75 of the top 100 coins by market value traded below both their 50-day and 200-day simple moving averages (SMAs), indicating across-the-board weakness in the digital asset market. "
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