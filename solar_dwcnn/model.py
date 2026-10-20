"""DW-CNN adapted from ../Dunglaibaibao/src/models/dw_cnn.py.

Same four convolution blocks, two Haar levels and additive encoder skips.
Haar is implemented by even/odd arithmetic (no dense diagonal convolutions).
A mask channel and an identity-initialized residual head support blind prediction.
"""
import tensorflow as tf
from tensorflow.keras import layers as L


@tf.keras.utils.register_keras_serializable(package="solar_dwcnn")
class HaarDWT(L.Layer):
    def build(self, shape):
        if shape[1] is not None and shape[1] % 2:
            raise ValueError("HaarDWT requires an even length")
        super().build(shape)

    def call(self, x):
        a, b = x[:, ::2, :], x[:, 1::2, :]
        s = tf.cast(2.0 ** -0.5, x.dtype)
        return tf.concat([(a + b) * s, (a - b) * s], axis=-1)

    def compute_output_shape(self, s):
        return (s[0], None if s[1] is None else s[1] // 2, s[2] * 2)


@tf.keras.utils.register_keras_serializable(package="solar_dwcnn")
class HaarIDWT(L.Layer):
    def build(self, shape):
        if shape[-1] % 2:
            raise ValueError("HaarIDWT requires an even channel count")
        super().build(shape)

    def call(self, x):
        lo, hi = tf.split(x, 2, axis=-1)
        s = tf.cast(2.0 ** -0.5, x.dtype)
        paired = tf.stack([(lo + hi) * s, (lo - hi) * s], axis=2)
        out = tf.reshape(paired, [tf.shape(x)[0], tf.shape(x)[1] * 2, x.shape[-1] // 2])
        out.set_shape(self.compute_output_shape(x.shape))
        return out

    def compute_output_shape(self, s):
        return (s[0], None if s[1] is None else s[1] * 2, s[2] // 2)


@tf.keras.utils.register_keras_serializable(package="solar_dwcnn")
class ValueChannel(L.Layer):
    def call(self, x):
        return tf.cast(x[..., :1], tf.float32)


def block(x, filters, final_filters=None):
    for _ in range(5):
        x = L.Conv1D(filters, 3, padding="same", activation="relu")(x)
    if final_filters is not None:
        x = L.Conv1D(final_filters, 3, padding="same", activation="relu")(x)
    return x


def DW_CNN(length=512, filters=64):
    if length < 16 or length % 4:
        raise ValueError("length must be >=16 and divisible by four")
    inputs = L.Input((length, 2), name="masked_value_and_mask")
    skip1 = block(inputs, filters)
    skip2 = block(HaarDWT()(skip1), filters)
    latent = block(HaarDWT()(skip2), filters, 2 * filters)
    dec = L.Add()([HaarIDWT()(latent), skip2])
    dec = block(dec, filters, 2 * filters)
    dec = L.Add()([HaarIDWT()(dec), skip1])
    for _ in range(2):
        dec = L.Conv1D(filters, 3, padding="same", activation="relu")(dec)
    correction = L.Conv1D(1, 3, padding="same", kernel_initializer="zeros",
                          bias_initializer="zeros", dtype="float32", name="correction")(dec)
    out = L.Add(dtype="float32", name="prediction")([ValueChannel(dtype="float32")(inputs), correction])
    return tf.keras.Model(inputs, out, name="Solar_DW_CNN")


@tf.keras.utils.register_keras_serializable(package="solar_dwcnn")
def masked_mse(y_true, y_pred):
    target, mask = y_true[..., :1], y_true[..., 1:2]
    error = tf.square(tf.cast(y_pred, tf.float32) - target) * mask
    return tf.reduce_sum(error, axis=[1, 2]) / tf.maximum(tf.reduce_sum(mask, axis=[1, 2]), 1.)
