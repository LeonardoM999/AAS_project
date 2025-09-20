import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Layer,
    Dense,
    Flatten,
    Conv2D,
    Rescaling,
    MaxPooling2D,
)

print("cuda in tf:", tf.test.is_built_with_cuda())
print(f'\n{tf.config.list_physical_devices("GPU")}\n')
