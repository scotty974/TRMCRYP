import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, num_heads, ff_dim, rate=0.4):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_size)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="gelu"),
            tf.keras.layers.Dense(embed_size),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    
    def call(self, inputs, training=False):
        attn_output = self.attn(inputs, inputs) 
        attn_output = self.dropout1(attn_output, training=training) 
        out1 = self.layernorm1(inputs + attn_output) 
        ffn_output = self.ffn(out1) 
        ffn_output = self.dropout2(ffn_output, training=training) 
        return self.layernorm2(out1 + ffn_output)