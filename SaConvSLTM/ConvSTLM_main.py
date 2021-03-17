from load_data import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from SaConvSLTM_main import *
def build_model():
    model = keras.Sequential(
        [
            keras.Input(
                shape=(None, 64, 64, 1)
            ),
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(
                filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
            ),
            layers.BatchNormalization(),
            layers.Conv3D(
                filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
            ),
        ]
    )
    # model.summary()
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(5))
    return model


def train(model, train_x, train_y):
    epochs = 80  # In practice, you would need hundreds of epochs.

    # # 在文件名中包含 epoch (使用 `str.format`)
    # checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    #
    # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    # print(ckpt)
    # if ckpt and ckpt.model_checkpoint_path:
    #     model = restore_from_latest(model)
    #
    # # 创建一个回调，每 5 个 epochs 保存模型的权重
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_path,
    #     verbose=1,
    #     save_weights_only=True,
    #     period=5)
    #
    # # 使用 `checkpoint_path` 格式保存权重
    # model.save_weights(checkpoint_path.format(epoch=0))
    model.fit(
        train_x,
        train_y,
        batch_size=8,
        epochs=epochs,
        # callbacks=[cp_callback],
        verbose=2,
        validation_split=0.1,
    )
    model.save_weights('saved_weight/')
    # print(seq.output_shape)
    # print(test_y.shape)
    return model


def predict(model, test_x, test_y):
    # loss, acc = model.evaluate(test_x, test_y, verbose=2)
    # print("accuracy: {:5.2f}%".format(100 * acc))
    prediction = model.predict(test_x)
    # turn [64, 64, 1] img to [64, 64] img. Otherwise may raise an error when plot
    prediction = np.squeeze(prediction, 4)  # shape = [batch_size, 10, 64, 64]
    save_as_image(prediction)
    stantard = np.squeeze(test_y, 4);
    save_as_image(stantard, 1)
    # for id in range(prediction.shape[0]):
    #     for t in range(prediction.shape[1]):
    #         plt.imshow(prediction[id, t])
    #         plt.show()


def restore_from_latest(model):
    checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(latest)
    model.load_weights(latest)
    return model


def main():
    print('new')
    path = download()
    data = load_data(path)
    train_x, train_y, test_x, test_y = split(data, 2950, 3000)  # (, 10, 64, 64)
    model = build_model()
    model = train(model, train_x, train_y)
    predict(model, test_x, test_y)

model = build_model()
Sa_train_test(model)