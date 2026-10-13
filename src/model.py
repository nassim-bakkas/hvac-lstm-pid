"""
LSTM model definition and training.

Architecture (unchanged from the original study):
    LSTM(100, return_sequences=True) -> Dropout(0.2)
    LSTM(50)                         -> Dropout(0.2)
    Dense(25) -> Dense(1)

An LSTM is used rather than a plain RNN because the vanishing-gradient
problem prevents a simple recurrent network from learning the ~24 h
periodicity of a daily temperature cycle. The gated cell state carries
information across that span.
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_model(look_back=48, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(100, input_shape=(look_back, 1), return_sequences=True))
    model.add(Dropout(0.2))  # reduce overfitting
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))  # intermediate dense layer
    model.add(Dense(1))  # output
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=learning_rate))
    return model


def train(model, x_train, y_train, validation_data=None, epochs=200, batch_size=16):
    """Train with early stopping.

    NOTE: the original study passed the *test* set as `validation_data` while
    early-stopping on `val_loss`, which lets the test set influence model
    selection. That is reproduced here by default so the published numbers can
    be regenerated, but it is a methodological flaw -- see the README.
    Pass a genuine held-out validation split for an honest evaluation.
    """
    early_stop = EarlyStopping(
        monitor="val_loss" if validation_data is not None else "loss",
        patience=10,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=[early_stop],
        verbose=1,
    )
    return history
