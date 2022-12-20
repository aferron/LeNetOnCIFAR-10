from keras import backend as K
from keras import regularizers
from keras import initializers
from keras.layers import AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten
from keras.models import Sequential

class LeNet:

    def build(self, option: int, numChannels: int, imgRows: int, imgCols: int, numClasses: int, activation: str, 
                weightsPath: str=None, verbose: bool=False, kernel_initializer: bool=False,
                bias_regularizer: bool=False, activity_regularizer: bool=False):

        # initialize the model
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)
        # if we are using "channels first", update the image shape
        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, imgRows, imgCols)
            if verbose:
                print("channels first")
    
        # define the Conv => activation => Pool layers
        model.add(Conv2D(
            filters=6, 
            kernel_size=(5, 5) if (option == 1) else (3, 3), 
            strides=(1, 1),
            padding = 'same' if (option == 3) else 'valid',
            activation=activation, 
            use_bias=True,
            kernel_initializer='glorot_normal' if kernel_initializer else 'glorot_uniform',
            bias_initializer=initializers.RandomNormal(mean=0, stddev=(1/(imgRows * imgCols * numChannels))),
            kernel_regularizer=regularizers.l2(1e-5),
            bias_regularizer=regularizers.l2(1e-7) if bias_regularizer else None,
            activity_regularizer=regularizers.l2(1e-5) if activity_regularizer else None,
            data_format=K.image_data_format(), 
            input_shape=inputShape))
        model.add(AveragePooling2D(
            pool_size=(2, 2), 
            strides=(1, 1) if (option == 3) else (2, 2), 
            padding='same' if (option ==3) else 'valid'))
        model.add(Conv2D(
            filters=16 if (option == 3) else 6, 
            kernel_size=(5, 5) if (option == 1) else (3, 3), 
            strides=(1, 1),
            padding='same' if (option == 3) else 'valid',
            activation=activation,
            use_bias=True,
            kernel_initializer='glorot_normal' if kernel_initializer else 'glorot_uniform',
            bias_initializer=initializers.RandomNormal(
                mean=0,
                stddev=(1/(imgRows * imgCols * numChannels))),
            # kernel_regularizer=regularizers.l1_l2(l1=13-5, l2=1e-4),
            kernel_regularizer=regularizers.l2(1e-5),
            bias_regularizer=regularizers.l2(1e-7) if bias_regularizer else None,
            activity_regularizer=regularizers.l2(1e-5) if activity_regularizer else None,
            data_format=K.image_data_format(), 
            input_shape=inputShape)) 
        model.add(AveragePooling2D(
            pool_size=(2, 2), 
            strides=(1, 1) if (option == 3) else (2, 2), 
            padding='same' if (option ==3) else 'valid'))
        if (option == 3):
            model.add(Conv2D(
                filters=6, 
                kernel_size=(3, 3), 
                strides=(1, 1),
                padding='same',
                activation=activation, 
                use_bias=True,
                kernel_initializer='glorot_normal' if kernel_initializer else 'glorot_uniform',
                bias_initializer=initializers.RandomNormal( mean=0,  stddev=(1/(imgRows * imgCols * numChannels))),
                kernel_regularizer=regularizers.l2(1e-5),
                bias_regularizer=regularizers.l2(1e-7) if bias_regularizer else None,
                activity_regularizer=regularizers.l2(1e-5) if activity_regularizer else None,
                data_format=K.image_data_format(), 
                input_shape=inputShape))
            model.add(AveragePooling2D(
                pool_size=(2, 2), 
                strides=(1, 1), 
                padding='same'))
            model.add(Conv2D(
                filters=6, 
                kernel_size=(3, 3), 
                strides=(1, 1),
                padding='same',
                activation=activation, 
                use_bias=True,
                kernel_initializer='glorot_normal' if kernel_initializer else 'glorot_uniform',
                bias_initializer=initializers.RandomNormal( mean=0,  stddev=(1/(imgRows * imgCols * numChannels))),
                kernel_regularizer=regularizers.l2(1e-5),
                bias_regularizer=regularizers.l2(1e-7) if bias_regularizer else None,
                activity_regularizer=regularizers.l2(1e-5) if activity_regularizer else None,
                data_format=K.image_data_format(), 
                input_shape=inputShape))
            model.add(AveragePooling2D(
                pool_size=(2, 2), 
                strides=(1, 1), 
                padding='same'))
            model.add(Conv2D(
                filters=16, 
                kernel_size=(3, 3), 
                strides=(1, 1),
                padding='same',
                activation=activation,
                use_bias=True,
                kernel_initializer='glorot_normal' if kernel_initializer else 'glorot_uniform',
                bias_initializer=initializers.RandomNormal(
                    mean=0, 
                    stddev=(1/(imgRows * imgCols * numChannels))),
                # kernel_regularizer=regularizers.l1_l2(l1=13-5, l2=1e-4),
                kernel_regularizer=regularizers.l2(1e-5),
                bias_regularizer=regularizers.l2(1e-7) if bias_regularizer else None,
                activity_regularizer=regularizers.l2(1e-5) if activity_regularizer else None,
                data_format=K.image_data_format(), 
                input_shape=inputShape)) 
            model.add(AveragePooling2D(
                pool_size=(2, 2), 
                strides=(1, 1), 
                padding='same'))

        # define the first FC => Activation layers
        model.add(Flatten())
        model.add(Dense(units=120, activation=activation))
        model.add(Dense(units=84, activation=activation))
        # define the second FC layer
        model.add(Dense(units=numClasses, activation='softmax'))
        # if a weights path is supplied (indicating that the model was pre-trained,
        # then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)
        # return the constructed network architecture
        return model