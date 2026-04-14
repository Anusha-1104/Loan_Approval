"""
src/model.py
LoanNet — Neural Network architecture for binary loan approval classification.
Python 3.11 | TensorFlow 2.15
"""

import tensorflow as tf
from tensorflow import keras


def build_loannet(
    input_dim:  int   = 17,
    dropout1:   float = 0.4,
    dropout2:   float = 0.3,
    l2_lambda:  float = 1e-4,
    lr:         float = 1e-3,
) -> keras.Model:
    """
    LoanNet: Feed-forward binary classifier for loan approval.

    Architecture
    ────────────
    Input  (input_dim,)
    Dense 256  + BatchNorm + Dropout(dropout1)   ← ReLU + L2
    Dense 128  + BatchNorm + Dropout(dropout2)   ← ReLU + L2
    Dense 64   + BatchNorm                       ← ReLU
    Dense 32                                     ← ReLU
    Dense 1    (sigmoid)                         ← P(approved)

    Args:
        input_dim  : number of input features
        dropout1   : dropout rate after first dense block
        dropout2   : dropout rate after second dense block
        l2_lambda  : L2 regularisation coefficient
        lr         : Adam initial learning rate

    Returns:
        Compiled keras.Model
    """
    reg = keras.regularizers.L2(l2_lambda)

    inp = keras.layers.Input(shape=(input_dim,), name="applicant_features")

    # Block 1
    x = keras.layers.Dense(256, activation="relu",
                            kernel_regularizer=reg,
                            name="dense_1")(inp)
    x = keras.layers.BatchNormalization(name="bn_1")(x)
    x = keras.layers.Dropout(dropout1, name="drop_1")(x)

    # Block 2
    x = keras.layers.Dense(128, activation="relu",
                            kernel_regularizer=reg,
                            name="dense_2")(x)
    x = keras.layers.BatchNormalization(name="bn_2")(x)
    x = keras.layers.Dropout(dropout2, name="drop_2")(x)

    # Block 3
    x = keras.layers.Dense(64, activation="relu",
                            kernel_regularizer=reg,
                            name="dense_3")(x)
    x = keras.layers.BatchNormalization(name="bn_3")(x)

    # Block 4
    x = keras.layers.Dense(32, activation="relu", name="dense_4")(x)

    # Output
    out = keras.layers.Dense(1, activation="sigmoid",
                              name="approval_prob")(x)

    model = keras.Model(inputs=inp, outputs=out, name="LoanNet")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )
    return model


def load_loannet(path: str = "models/loannet_best.h5") -> keras.Model:
    """Load a saved LoanNet model from disk."""
    return keras.models.load_model(path)


if __name__ == "__main__":
    model = build_loannet(input_dim=17)
    model.summary()
    print(f"\nTotal params: {model.count_params():,}")
