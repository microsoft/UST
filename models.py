"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for uncertainty-aware self-training for few-shot learning.
"""

import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model


def construct_teacher(TFModel, Config, pt_teacher_checkpoint, max_seq_length, classes, dense_dropout=0.5, attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2):

    config = Config.from_pretrained(pt_teacher_checkpoint, num_labels=classes)
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    config.hidden_dropout_prob = hidden_dropout_prob
    encoder = TFModel.from_pretrained(pt_teacher_checkpoint, config=config, from_pt=True, name="teacher")

    input_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_seq_length,), dtype=tf.int32, name="attention_mask")
    token_type_ids = Input(shape=(max_seq_length,), dtype=tf.int32, name="token_type_ids")

    output = encoder(input_ids, token_type_ids=token_type_ids,  attention_mask=attention_mask)
    output = Dropout(dense_dropout)(output[0][:,0])
    output = Dense(classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range))(output)
    model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=output)
    return model
