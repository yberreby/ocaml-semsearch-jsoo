#!/usr/bin/env python3

from transformers import TFAutoModel, AutoTokenizer, TFBertTokenizer
import tensorflow as tf
import numpy as np


class TFSentenceTransformer(tf.keras.Model):
    def __init__(self, model_name_or_path, **kwargs):
        super(TFSentenceTransformer, self).__init__()
        # loads transformers model
        self.model = TFAutoModel.from_pretrained(model_name_or_path, **kwargs)

    def call(self, inputs, normalize=True):
        # runs model on inputs
        model_output = self.model(inputs)
        # Perform pooling. In this case, mean pooling.
        embeddings = self.mean_pooling(model_output, inputs["attention_mask"])
        # normalizes the embeddings if wanted
        if normalize:
            embeddings = self.normalize(embeddings)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = tf.cast(
            tf.broadcast_to(
                tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)
            ),
            tf.float32,
        )
        return tf.math.reduce_sum(
            token_embeddings * input_mask_expanded, axis=1
        ) / tf.clip_by_value(
            tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max
        )

    def normalize(self, embeddings):
        embeddings, _ = tf.linalg.normalize(embeddings, 2, axis=1)
        return embeddings
