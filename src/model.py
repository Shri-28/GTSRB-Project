"""
model.py — ResNet50 classifier for GTSRB traffic sign recognition.

Design decisions (see 00-Proposal for full justification):

- ResNet50 pretrained on ImageNet, not trained from scratch. GTSRB has
  ~31k training images across 43 classes — enough to fine-tune, not
  enough to train a ResNet50 from random init without overfitting.
  ResNet50 is also the common benchmark backbone, and every attention
  module under consideration (CBAM / SE / ECA) has a documented
  reference implementation for ResNet bottleneck blocks, so swapping
  one in later doesn't require re-deriving the architecture.

- Two-phase training (freeze -> fine-tune), identical across ALL five
  ablation runs. Phase 1 trains only the new classification head while
  the backbone is frozen, so the randomly-initialized head doesn't
  push large, noisy gradients into the pretrained weights. Phase 2
  unfreezes the backbone and fine-tunes end-to-end at a much lower
  learning rate. Keeping this schedule fixed across runs is what makes
  the ablation valid: only the run-specific ingredient (loss weighting,
  augmentation, crop, attention) should differ between runs.

- Class weighting is applied via `class_weight` in model.fit(), not
  baked into a custom loss function. That keeps the loss function
  itself identical across all runs, and turning the imbalance fix
  on/off for Run 1 vs Run 2 becomes a single argument rather than a
  code branch.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

IMG_SIZE = 64
NUM_CLASSES = 43


def build_resnet50_model(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    num_classes=NUM_CLASSES,
    freeze_base=True,
    weights="imagenet",
    dropout=0.3,
    augmentation=None
):
    """
    Build a ResNet50 + classification head for GTSRB.

    Returns (model, base_model) — base_model is returned separately so
    training scripts can flip `base_model.trainable` for phase 2
    without having to search the full model graph for it.

    Input-resolution note: ResNet50 downsamples by 32x total (stem: /4,
    then four stages at /2 each). At 64x64 input, the feature map
    feeding GlobalAveragePooling is only 2x2xC. That's coarse — it's a
    known trade-off of running ResNet50 at GTSRB's native-ish
    resolution instead of upscaling to 224x224, and it's one plausible
    reason the Run 1-3 baselines still confuse visually similar classes
    (e.g. speed limit digits) even before crop/attention are added.
    Worth calling out explicitly when interpreting early confusion
    matrices, rather than jumping straight to "we need attention."
    """
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling=None,
    )
    base_model.trainable = not freeze_base

    inputs = keras.Input(shape=input_shape, name="image")
    # Data pipeline (01_data_pipeline.ipynb) stores images as float32 in
    # [0, 1]. keras.applications.resnet50.preprocess_input expects
    # [0, 255] inputs (it does caffe-style BGR mean subtraction), so we
    # rescale before handing off. VERIFY this against the actual
    # X_train.npy value range before trusting it — see the sanity-check
    # cell in 02_training.ipynb.
    x = inputs
    if augmentation is not None : 
        x = augmentation(x)

    x = layers.Rescaling(255.0, name="to_0_255")(x)
    x = keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=not freeze_base)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="resnet50_gtsrb")
    return model, base_model


def compile_model(model, learning_rate=1e-3):
    """
    Plain sparse categorical crossentropy — used for every run.
    Class weighting (Run 2+) is passed to model.fit(class_weight=...)
    by the training script, not applied here.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    if keras.mixed_precision.global_policy().name == "mixed_float16":
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_for_finetune(model, base_model, learning_rate=1e-5):
    """
    Phase 2: unfreeze the backbone and recompile at a lower LR.
    Recompiling is required in Keras for a `trainable` flip to take
    effect on the optimizer's variable list.
    """
    base_model.trainable = True
    compile_model(model, learning_rate=learning_rate)
    return model


def get_callbacks(checkpoint_path, patience=5, use_wandb=False):
    """
    Standard callback stack, identical across runs so training dynamics
    (not the schedule) are what differ between ablation runs.
    """
    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True
        ),
    ]
    if use_wandb:
        from wandb.integration.keras import WandbMetricsLogger

        cb.append(WandbMetricsLogger())
    return cb