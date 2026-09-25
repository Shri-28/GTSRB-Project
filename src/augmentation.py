"""
augmentation.py — training-time data augmentation for GTSRB.

Design notes:

- Augmentation is a keras.Sequential of layers, meant to be attached as the
  FIRST layer inside the model graph (see model.py's `augmentation` argument),
  not applied in the tf.data pipeline. Keras augmentation layers automatically
  no-op when the model is called with training=False, so this stays active
  during model.fit() and automatically skips itself during model.predict()/
  evaluate() — no manual on/off switching needed.

- RandomFlip("horizontal") is included but worth flagging: GTSRB has
  direction-sensitive classes (turn-left/turn-right arrows, keep-left/
  keep-right signs). A horizontal flip turns one of those into a different,
  wrong class rather than a helpful augmentation of the same class. Worth
  checking your class list before trusting this blindly — if any of the 43
  classes are directional, consider dropping RandomFlip or restricting it
  to a subset of classes.
"""

import tensorflow as tf
from tensorflow import keras


class RandomErasing(keras.layers.Layer):
    """
    Randomly blacks out a rectangle in the image.
    Simulates partial occlusion — a branch or vehicle partly covering the sign.
    Only active during training (training=True); a no-op otherwise.
    """

    def __init__(self, probability=0.5, min_area=0.02, max_area=0.15, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area

    def call(self, images, training=None):
        if not training:
            return images

        def erase(imgs):
            shape = tf.shape(imgs)
            h, w = shape[1], shape[2]
            area = tf.cast(h * w, tf.float32)
            erase_area = tf.random.uniform([], self.min_area, self.max_area) * area
            aspect = tf.random.uniform([], 0.5, 2.0)
            eh = tf.minimum(tf.cast(tf.sqrt(erase_area / aspect), tf.int32), h)
            ew = tf.minimum(tf.cast(tf.sqrt(erase_area * aspect), tf.int32), w)
            top = tf.random.uniform([], 0, h - eh + 1, dtype=tf.int32)
            left = tf.random.uniform([], 0, w - ew + 1, dtype=tf.int32)
            patch = tf.zeros((eh, ew, 1), dtype=imgs.dtype)
            padded = tf.pad(
                patch,
                [[top, h - top - eh], [left, w - left - ew], [0, 0]],
                constant_values=1.0,
            )
            return imgs * tf.expand_dims(padded, 0)

        should_apply = tf.random.uniform([]) < self.probability
        return tf.cond(should_apply, lambda: erase(images), lambda: images)


def build_augmentation_pipeline():
    """
    Returns a keras.Sequential of augmentation layers, ready to pass as
    `augmentation=` into build_resnet50_model() in model.py.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomTranslation(0.05, 0.05),
            keras.layers.RandomBrightness(0.2, value_range=(0.0, 1.0)),
            keras.layers.RandomContrast(0.2),
            RandomErasing(probability=0.5),
        ],
        name="augmentation",
    )