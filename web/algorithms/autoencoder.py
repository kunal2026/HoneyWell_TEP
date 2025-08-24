
import numpy as np

def available():
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except Exception:
        return False

def build_autoencoder(input_dim, bottleneck=8):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.Dense(64, activation='relu')(x)
    z = layers.Dense(bottleneck, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(z)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(input_dim, activation=None)(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

def score(X, epochs=30, bottleneck=8, batch_size=128, random_state=42):
    if not available():
        raise ImportError('TensorFlow is required for the Autoencoder. Install with `pip install tensorflow`.')
    import numpy as np
    import tensorflow as tf
    tf.random.set_seed(random_state)
    model = build_autoencoder(X.shape[1], bottleneck=bottleneck)
    # train quietly
    model.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True)
    Xhat = model.predict(X, verbose=0)
    # reconstruction error per row (sum across features) normalized
    per_feature_err = ((X - Xhat)**2)
    row_err = per_feature_err.sum(axis=1)
    # also return per-feature contributions matrix for later use
    return row_err, per_feature_err
