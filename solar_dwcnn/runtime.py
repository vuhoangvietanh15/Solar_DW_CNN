import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf


def configure(allow_cpu=False, mixed=False, seed=42):
    tf.keras.utils.set_random_seed(seed)
    devices = tf.config.list_physical_devices('GPU')
    for d in devices:
        tf.config.experimental.set_memory_growth(d, True)
    if not devices and not allow_cpu:
        raise RuntimeError('TensorFlow sees no GPU. Check environment; use --allow-cpu only deliberately.')
    tf.keras.mixed_precision.set_global_policy('mixed_float16' if mixed and devices else 'float32')
    print('TensorFlow:', tf.__version__, 'GPU:', devices,
          'precision:', tf.keras.mixed_precision.global_policy().name, flush=True)
    return devices
