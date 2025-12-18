import tensorflow as tf
from tensorflow.keras.layers import Embedding
from blocks.TransformerBlock import TransformerBlock

max_len = 200
class Transformers(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, num_heads, ff_dim, rate=0.1):
        super(Transformers, self).__init__()
        self.embedding = Embedding(vocab_size, embed_size)
        self.pos_emb = Embedding(max_len, embed_size) 
        self.trans_bock = TransformerBlock(embed_size, num_heads, ff_dim, rate)
        self.final_layer = tf.keras.layers.Dense(6, activation="sigmoid")
    
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        x = x + positions
        x = self.trans_bock(x, training=training)
        x = x[:, -1, :]
        return self.final_layer(x)