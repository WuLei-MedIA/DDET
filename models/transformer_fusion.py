

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class VectorSequenceTransformerEncoder(layers.Layer):
    def __init__(self, num_layers, num_heads, proj_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.proj_dim = proj_dim
        self.dropout_rate = dropout_rate
        self.seq_len = 8

        self.projection = layers.Dense(proj_dim)

        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(self.seq_len, proj_dim),
            initializer="zeros",
            trainable=True
        )

        self.mha_layers = [
            layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=proj_dim // num_heads,
                dropout=dropout_rate
            ) for _ in range(num_layers)
        ]
        self.norm_layers = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers * 2)]
        self.dropout_layers = [layers.Dropout(dropout_rate) for _ in range(num_layers)]
        self.ffn_layers = [
            tf.keras.Sequential([
                layers.Dense(proj_dim * 4, activation="gelu"),
                layers.Dropout(dropout_rate),
                layers.Dense(proj_dim)
            ]) for _ in range(num_layers)
        ]

    def call(self, inputs, training=None):
        B = tf.shape(inputs)[0]
        total_dim = tf.shape(inputs)[-1]

        dim_per_token = total_dim // self.seq_len
        x = tf.reshape(inputs, [B, self.seq_len, dim_per_token])

        x = self.projection(x)

        pos_emb = tf.expand_dims(self.pos_embedding, axis=0)
        pos_emb = tf.tile(pos_emb, [B, 1, 1])
        x = x + pos_emb

        for i in range(self.num_layers):
            attn_output = self.mha_layers[i](x, x)
            attn_output = self.dropout_layers[i](attn_output, training=training)
            x = self.norm_layers[i*2](x + attn_output)

            ffn_output = self.ffn_layers[i](x)
            x = self.norm_layers[i*2 + 1](x + ffn_output)

        return x
