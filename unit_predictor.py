import numpy as np
import cPickle as pickle
import keras.models
import sklearn.neighbors
import sklearn.externals
import pianoroll_utils
import h5py

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

class UnitSelectorV2(UnitPredictor):
    def __init__(self):
        UnitPredictor.__init__(self)
        self.MIN_PITCH = 13
        self.MAX_PITCH = 108
        self.NUM_PITCHES = self.MAX_PITCH - self.MIN_PITCH + 1
        # Load up the kNN model along with the units used to learn the model
        KNN_MODEL_FILE = "./models/vae_v7_unit_selector_knn.pkl"
        self.knn_model, UNITS_FILE = sklearn.externals.joblib.load(KNN_MODEL_FILE)
        f = h5py.File(UNITS_FILE, 'r')
        self.units = f['units_train']
        # Load up the encoder model
        ENCODER_MODEL_FILE = "./models/vae_v7_encoder.h5"
        self.encoder = keras.models.load_model(ENCODER_MODEL_FILE)
        return

    def get_comp_pianoroll(self, input_pianoroll):
        """
        Given a input pianoroll with shape [NUM_PITCHES, NUM_TICKS],
        return an accompanying pianoroll with equivalent shape.
        """
        # Get input_pianoroll into the right shape
        input_pianoroll = pianoroll_utils.crop_pianoroll(input_pianoroll, self.MIN_PITCH, self.MAX_PITCH)
        # Get encoding of the input
        input_pianoroll = input_pianoroll[np.newaxis, ..., np.newaxis]
        input_encoding = self.encoder.predict(input_pianoroll)
        # Retrieve closest neighbor
        knn_index = self.knn_model.kneighbors(input_encoding, return_distance = False)[0][0]
        knn_pianoroll = self.units[knn_index].squeeze()
        # Pad the pianoroll from 88 to 128 keys
        knn_pianoroll = pianoroll_utils.pad_pianoroll(knn_pianoroll, self.MIN_PITCH, self.MAX_PITCH)
        return knn_pianoroll

class UnitAutoencoder(UnitPredictor):
    def __init__(self):
        UnitPredictor.__init__(self)
        AUTOENCODER_MODEL_FILE = "./models/autoencoder_v4_input_comp.h5"
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
        # Quantize the output
        output_pianoroll[output_pianoroll < 10] = 0
        output_pianoroll[output_pianoroll > 0] = 100
        return output_pianoroll

class UnitVariationalAutoencoder(UnitPredictor):
    def __init__(self):
        UnitPredictor.__init__(self)
        ENCODER_MODEL_FILE = "./models/vae_v1_encoder.h5"
        DECODER_MODEL_FILE = "./models/vae_v1_generator.h5"
        # Load up the autoencoder model
        latent_dim = 2400
        epsilon_std = 1.0

        self.encoder = keras.models.load_model(ENCODER_MODEL_FILE,
            custom_objects={'latent_dim': latent_dim, 
                            'epsilon_std': epsilon_std})
        self.decoder = keras.models.load_model(DECODER_MODEL_FILE,
            custom_objects={'latent_dim': latent_dim, 
                            'epsilon_std': epsilon_std})
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
        # autoencoder_output = self.encoder.predict(input_pianoroll) # (1, 128, 96, 1)
        z = self.encoder.predict(input_pianoroll)
        autoencoder_output = self.decoder.predict(z)

        output_pianoroll = autoencoder_output[0].reshape(self.NUM_PITCHES, self.NUM_TICKS) * 127
        # Quantize the output
        output_pianoroll[output_pianoroll < 10] = 0
        output_pianoroll = np.clip(output_pianoroll * 2, 0, 127)
        return output_pianoroll

class UnitAccompanier(UnitPredictor):
    def __init__(self):
        UnitPredictor.__init__(self)
        # Music shape
        self.MIN_PITCH = 21 # A-1 (MIDI 21)
        self.MAX_PITCH = 108 # C7 (MIDI 108)
        self.NUM_PITCHES = self.MAX_PITCH - self.MIN_PITCH + 1
        # Load up all our Keras models
        latent_dim = 10
        epsilon_std = 1.0
        ENCODER_MODEL_FILE = './models/vae_v4_encoder.h5'
        DECODER_MODEL_FILE = './models/vae_v4_generator.h5'
        RNN_MODEL_FILE = './models/rlstm_v3.h5'
        self.encoder = keras.models.load_model(ENCODER_MODEL_FILE, 
            custom_objects={'latent_dim': latent_dim, 'epsilon_std': epsilon_std})
        self.decoder = keras.models.load_model(DECODER_MODEL_FILE, 
            custom_objects={'latent_dim': latent_dim, 'epsilon_std': epsilon_std})
        self.rnn = keras.models.load_model(RNN_MODEL_FILE)
        # Prepare the fixed memory
        self.WINDOW_LENGTH = 4
        self.x_input_embed = np.zeros((self.WINDOW_LENGTH, latent_dim))
        self.x_comp_embed = np.zeros((self.WINDOW_LENGTH, latent_dim))
        return

    def get_comp_pianoroll(self, input_pianoroll):
        """
        Given a input pianoroll with shape [NUM_PITCHES, NUM_TICKS],
        return an accompanying pianoroll with equivalent shape.
        """
        # Normalize input_pianoroll
        input_pianoroll = input_pianoroll / 127.
        # Resize input_pianoroll from 128 to 88 keys
        input_pianoroll = pianoroll_utils.crop_pianoroll(input_pianoroll.T,
            self.MIN_PITCH, self.MAX_PITCH).T

        # Get encoding of the input
        input_pianoroll = input_pianoroll.reshape(1, self.NUM_PITCHES, self.NUM_TICKS, 1)
        input_embed = self.encoder.predict(input_pianoroll) # (1, 10)
        assert(input_embed.shape == (1, 10))
        # Append new input to past-inputs window
        self.x_input_embed = np.concatenate([self.x_input_embed[1:], input_embed], axis=0)
        assert(self.x_input_embed.shape == (self.WINDOW_LENGTH, 10))
        
        # Get prediction of next comp embedding
        next_comp_embed = self.rnn.predict([np.array([self.x_input_embed]), # (1, 10)
                                            np.array([self.x_comp_embed]) ])
        assert(next_comp_embed.shape == (1, 10))
        
        # Decode next comp embedding
        next_comp = self.decoder.predict(next_comp_embed) # (1, NUM_PITCHES, NUM_TICKS, 1)
        output_pianoroll = next_comp[0].reshape(self.NUM_PITCHES, self.NUM_TICKS) * 127
        # Quantize the output
        output_pianoroll[output_pianoroll < 10] = 0
        output_pianoroll[output_pianoroll > 0] = 100
        # Pad the pianoroll from 88 to 128 keys
        output_pianoroll = pianoroll_utils.pad_pianoroll(output_pianoroll.T,
            self.MIN_PITCH, self.MAX_PITCH).T

        # Append new comp to past-comps window
        self.x_comp_embed = np.concatenate([self.x_comp_embed[1:], next_comp_embed], axis=0)
        assert(self.x_comp_embed.shape == (self.WINDOW_LENGTH, 10))
        return output_pianoroll
