from keras import backend as K
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lenet import LeNet

class Experiment:
    def __init__(
        self,
        learning_rate: float,
        activation_function: str,
        loss_function: str,
        normalize: bool,
        epochs: int,
        batch_size: int,
        load_model: bool,
        weights_path: str,
        save_model: bool,
        option: int
    ):
        self.__learning_rate = learning_rate
        self.__activation_function = activation_function
        self.__loss_function = loss_function
        self.__normalize = normalize
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__load_model = load_model
        self.__weights_path = weights_path
        self.__save_model = save_model
        self.__option = option

    def run(self):
        ((trainData, trainLabels), (testData, testLabels)) = cifar10.load_data()

        if K.image_data_format() == "channels_first":
            trainData = trainData.reshape((trainData.shape[0], 3, 32, 32))
            testData = testData.reshape((testData.shape[0], 3, 32, 32))
        else:
            trainData = trainData.reshape((trainData.shape[0], 32, 32, 3))
            testData = testData.reshape((testData.shape[0], 32, 32, 3))

        if self.__normalize:
            trainData = trainData.astype("float32") / 255.0
            testData = testData.astype("float32") / 255.0
        else:
            trainData = trainData.astype("float32")
            testData = testData.astype("float32")

        trainLabels = np_utils.to_categorical(trainLabels, 10)
        testLabels = np_utils.to_categorical(testLabels, 10)

        opt = tf.keras.optimizers.SGD(lr=self.__learning_rate)

        model = LeNet.build(option=self.__option, numChannels=3, imgRows=32, imgCols=32, numClasses=10, 
                        activation=self.__activation_function, weightsPath=self.__weights_path)

        model.compile(loss=self.__loss_function, optimizer=opt, metrics=["accuracy"])

        if self.__load_model is False:
            history = model.fit(
                    trainData, 
                    trainLabels, 
                    batch_size=self.__batch_size, 
                    epochs=self.__epochs, 
                    verbose=2,
                    validation_split=0.2,
                    shuffle=True,
                    validation_freq=1)
            (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, 
                                                verbose=1)
            print('LR {}  BATCH {}  ACTIVATION {}  LOSS {} EPOCHS {}'.format(
                    self.__learning_rate, self.__batch_size, self.__activation_function, self.__loss_function, self.__epochs))
            print("Accuracy: {:.4f}%".format(accuracy))

            # check to see if the model should be saved to file
            if self.__save_model is True:
                    model.save_weights(self.__weights_path, overwrite=True)

            self.__plot(history)

    def __plot(self, history):
        print(history.history.keys())
        epoch_range = np.arange(self.__epochs)
        plt.plot(epoch_range, history.history['accuracy'], scaley=False)
        plt.plot(epoch_range, history.history['val_accuracy'], scaley=False)
        plt.xlabel('Epochs\nLR: {}  Activation: {}  Loss: {} Epochs: {}  Accuracy: {:.4f}'.format(
            self.__learning_rate, self.__activation_function, self.__loss_function, self.__epochs, self.__accuracy))
        plt.ylabel('Accuracy')
        plt.legend(['train', 'test'])
        plt.show()
