import numpy as np
import cPickle as pickle
import keras.models
import sklearn.neighbors
import sklearn.externals

class UnitPredictor:
    def __init__(self):
        # Other properties
        self.NUM_PITCHES = 128
        self.NUM_TICKS = 96
        return

    def get_comp_pianoroll(self, input_pianoroll):
        """
        Placeholder function, currently returns the same input.
        Should override in future classes to return a new pianoroll.
        """
        return input_pianoroll

class UnitSelector(UnitPredictor):
    def __init__(self):
        UnitPredictor.__init__(self)
        KNN_MODEL_FILE = "./pickle_jar/unit_selector_knn.pkl"
        ENCODER_MODEL_FILE = "./models/encoder_v2_input_input.h5"
        # Load up the kNN model along with the units used to learn the model
        self.units = {}
        self.knn_model, self.units["input"], self.units["comp"] = sklearn.externals.joblib.load(KNN_MODEL_FILE)
        # Load up the encoder model
        self.encoder = keras.models.load_model(ENCODER_MODEL_FILE)
        return
    
    def get_flattened_encodings(self, inputs, encoder):
        """
        Given an input matrix of shape (M, 128, 96, 1) and a trained encoder model,
        run each M pianorolls through the encoder and return an (M, F) matrix 
        where F is the length of the FLATTENED embedding layer.
        """
        assert inputs.shape[1] == self.NUM_PITCHES
        assert inputs.shape[2] == self.NUM_TICKS
        assert inputs.shape[3] == 1
        
        encodings = encoder.predict(inputs)
        flat_encodings = encodings.reshape(encodings.shape[0], -1)
        return flat_encodings

    def get_comp_pianoroll(self, input_pianoroll):
        """
        Given a input pianoroll with shape [NUM_PITCHES, NUM_TICKS],
        return an accompanying pianoroll with equivalent shape.
        """
        # Normalize input_pianoroll
        input_pianoroll = input_pianoroll / 127.
        # Get encoding of the input
        input_pianoroll = input_pianoroll.reshape(1, 128, 96, 1)
        input_encoding = self.get_flattened_encodings(input_pianoroll, self.encoder)
        # Prediction
        knn_index = self.knn_model.kneighbors(input_encoding, return_distance = False)[0][0]
        # Retrieve pianoroll
        comp_pianoroll = self.units["input"][knn_index].reshape(self.NUM_PITCHES, self.NUM_TICKS) * 127
        return comp_pianoroll

class UnitAutoencoder(UnitPredictor):
    def __init__(self):
        UnitPredictor.__init__(self)
        AUTOENCODER_MODEL_FILE = "./models/autoencoder_v4.h5"
        # Load up the autoencoder model
        self.autoencoder = keras.models.load_model(AUTOENCODER_MODEL_FILE)
        return

    def get_comp_pianoroll(self, input_pianoroll):
        """
        Given a input pianoroll with shape [NUM_PITCHES, NUM_TICKS],
        return an accompanying pianoroll with equivalent shape.
        """
        # Normalize input_pianoroll
        input_pianoroll = input_pianoroll / 127.
        # Get encoding of the input
        input_pianoroll = input_pianoroll.reshape(1, self.NUM_PITCHES, self.NUM_TICKS, 1)
        autoencoder_output = self.autoencoder.predict(input_pianoroll) # (1, 128, 96, 1)
        output_pianoroll = autoencoder_output[0].reshape(self.NUM_PITCHES, self.NUM_TICKS) * 127
        # Artificially boost the output :(
        output_pianoroll = np.clip(output_pianoroll * 3, 0, 127)
        return output_pianoroll
