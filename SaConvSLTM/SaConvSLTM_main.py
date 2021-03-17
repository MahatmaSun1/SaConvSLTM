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
    # plot the structure of model
    # keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    # output the summary of model
    # model.summary()
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.05))
    return model

# call this func to load data and test model
def Sa_train_test(model):
    # process data
    path = download()
    data = load_data(path)
    train_x, train_y, test_x, test_y = split(data, 2950, 3000)  # shape=(, 10, 64, 64), just for test. Modify (2950, 3000) to about 10,000 in practice, such as (9800, 10000)
    # build model 
    model = Sa_build_model()

    epochs = 80  # shape(, 10, 64, 64), just for test. should be close to 10000 in practice, such as (9800, 10000)
    model.fit(
        train_x,
        train_y,
        batch_size=8,
        epochs=epochs,
        verbose=2,
        validation_split=0.1,
    )
    # save trained weight
    model.save_weights('sa_saved_weight/')
    # make prediction
    prediction = model.predict(test_x)
    # turn [64, 64, 1] img to [64, 64] img. Otherwise may raise an error when plot
    prediction = np.squeeze(prediction, 4)  # shape = [batch_size, 10, 64, 64]
    # save result as photoes
    save_as_image(prediction)
    # save the standard result, that is, test_y, as photoes
    # stantard = np.squeeze(test_y, 4);
    # save_as_image(stantard, 1)
    return model
