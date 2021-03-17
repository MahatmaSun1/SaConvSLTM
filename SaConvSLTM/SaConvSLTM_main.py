from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from SaConvLSTM import *
from load_data import *


def Sa_build_model():
    filters = 40
    model = keras.Sequential([
        keras.Input(
            shape=(None, 64, 64, 1)
        ),
        SaConvLSTM2D(
            filters=filters, kernel_size=(3, 3), padding="same", return_sequences=True
        ), tf.keras.layers.BatchNormalization(),
        SaConvLSTM2D(
            filters=filters, kernel_size=(3, 3), padding="same", return_sequences=True
        ), tf.keras.layers.BatchNormalization(),
        SaConvLSTM2D(
            filters=filters, kernel_size=(3, 3), padding="same", return_sequences=True
        ), tf.keras.layers.BatchNormalization(),
        SaConvLSTM2D(
            filters=filters, kernel_size=(3, 3), padding="same", return_sequences=True
        ), tf.keras.layers.BatchNormalization(),
        layers.Conv3D(
            filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
        ),
    ]
    )
    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    # model.summary()
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.05))
    return model


def Sa_train_test(model):
    path = download()
    data = load_data(path)
    train_x, train_y, test_x, test_y = split(data, 2950, 3000)  # (, 10, 64, 64)
    model = Sa_build_model()

    # print(model.get_losses_for(train_x).shape)
    epochs = 2  # In practice, you would need hundreds of epochs.
    model.fit(
        train_x,
        train_y,
        batch_size=8,
        epochs=epochs,
        # callbacks=[cp_callback],
        verbose=2,
        validation_split=0.1,
    )
    # save trained weight
    model.save_weights('sa_saved_weight/')
    prediction = model.predict(test_x)
    # turn [64, 64, 1] img to [64, 64] img. Otherwise may raise an error when plot
    prediction = np.squeeze(prediction, 4)  # shape = [batch_size, 10, 64, 64]
    save_as_image(prediction)
    # stantard = np.squeeze(test_y, 4);
    # save_as_image(stantard, 1)
    return model
