

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils.Res50_art import resnet50_art
from utils.pvp import resnet50_pvp
from utils.TransEncoder import VectorSequenceTransformerEncoder

# -----------------------------
IMG_SIZE = (D, H, W)
INPUT_SHAPE = IMG_SIZE + (1,)
NUM_TRANS_LAYERS = nums
NUM_TRANS_HEADS = heads
TRANS_DROPOUT = 0.1
TRANS_PROJ_DIM = dims
# -----------------------------

def ddet(return_features=False):
    arterial_input = layers.Input(shape=INPUT_SHAPE, name="arterial_phase")
    portal_input = layers.Input(shape=INPUT_SHAPE, name="portal_phase")

    encoder_arterial = resnet50_art(INPUT_SHAPE)
    encoder_portal = resnet50_pvp(INPUT_SHAPE)

    AF_s2, AF_s3, AF_s4, AF_s5, AF_s2_conv, AF_s3_conv, AF_s4_conv, AF_s5_conv = encoder_arterial(arterial_input)
    PF_s2, PF_s3, PF_s4, PF_s5, PF_s2_conv, PF_s3_conv, PF_s4_conv, PF_s5_conv = encoder_portal(portal_input)

    def diff_softmask(af, pf):
        diff = af - pf
        return diff * tf.keras.activations.sigmoid(diff * 5.0)

    diff_s2 = diff_softmask(AF_s2, PF_s2)
    diff_s3 = diff_softmask(AF_s3, PF_s3)
    diff_s4 = diff_softmask(AF_s4, PF_s4)
    diff_s5 = diff_softmask(AF_s5, PF_s5)

    sequence = layers.Concatenate(axis=1)(
        [AF_s2, PF_s2, diff_s2,
         AF_s3, PF_s3, diff_s3,
         AF_s4, PF_s4, diff_s4,
         AF_s5, PF_s5, diff_s5]
    )

    transformer = VectorSequenceTransformerEncoder(
        num_layers=NUM_TRANS_LAYERS,
        num_heads=NUM_TRANS_HEADS,
        proj_dim=TRANS_PROJ_DIM,
        dropout_rate=TRANS_DROPOUT
    )
    encoded_sequence = transformer(sequence)

    pooled = layers.GlobalAveragePooling1D()(encoded_sequence)

    x = layers.Dropout(0.5)(pooled)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    scores = layers.Dense(1)(x)

    out_puts = [outputs, scores, sequence]
    if return_features:
        out_puts.extend([
            AF_s5_conv, PF_s5_conv,
            AF_s5, PF_s5,
            AF_s4_conv, PF_s4_conv,
            AF_s4, PF_s4,
            AF_s3_conv, PF_s3_conv,
            AF_s3, PF_s3,
            AF_s2_conv, PF_s2_conv,
            AF_s2, PF_s2,
        ])

    model = keras.Model(inputs=[arterial_input, portal_input],
                        outputs=out_puts,)
    return model