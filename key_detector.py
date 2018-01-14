"""
Key detector based on bag-of-words histogram detection
"""
import sys
import time
import mido
import numpy as np
import sklearn.neighbors
import sklearn.externals
import matplotlib.pyplot as plt

NOTE2NAME = {0:"C", 1:"C#", 2:"D", 3:"D#", 4:"E", 5:"F", \
    6:"F#", 7:"G", 8:"G#", 9:"A", 10:"A#", 11:"B"}
NAME2NOTE = {"C":0, "C#":1, "D":2, "D#":3, "E":4, "F":5, \
    "F#":6, "G":7, "G#":8, "A":9, "A#":10, "B":11}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NUM_NOTES = len(NOTE2NAME)

class KeyDetector(object):
    def __init__(self):
        self.data = np.zeros(0) # X
        self.labels = np.zeros(0) # y
        self.knn_model = None # Classification model
        return

    def info(self):
        text = """
        KeyDetector can be used for training/predicting a classifier
        for the musical key of a MIDI file / sequence of MIDI notes.
        
        For command line use: 
        
        Run the trainer with 2 arguments eg.
        'python key_detector.py train path/to/midi/file.mid <key>'
        where key is any of C, C#, D, D#... etc.
        
        Classification of a midi file can be done with
        'python key_detector.py predict path/to/midi/file.mid'
        """
        return text

    ## Not necessary to transpose the entire file, just transpose the histogram!
    # def transpose_midi_file(self, midifile, ori_note, target_note):
    #     """
    #     Naive transposition using a blanket addition of interval to all note messages.
    #     """
    #     interval = target_note - ori_note
    #     new_midifile = mido.MidiFile()
    #     track = mido.MidiTrack()
    #     new_midifile.tracks.append(track)
    #     for msg in mido.MidiFile(midifile):
    #         new_msg = msg.copy()
    #         if not msg.is_meta:
    #             new_msg = new_msg.copy(note = new_msg.note + interval)
    #         # Copy over the (transposed) messages from old file to new file
    #         track.append(new_msg)
    #     return new_midifile

    def midi_to_note(self, note_number):
        normalized_note_number = note_number % 12
        return normalized_note_number

    def normalize(self, hist):
        largest_value = max(hist)
        norm_hist = hist / largest_value
        return norm_hist

    def load_model(self, filename='knn_model.pkl'):
        self.knn_model, self.labels = sklearn.externals.joblib.load(filename)
        return

    def file2histogram(self, midifile):
        """
        Given a path to a MIDI file, return a histogram of note probabilities.
        """
        msglist = mido.MidiFile(midifile)
        return self.get_histogram(msglist)

    def get_histogram(self, msglist):
        """
        Given a list of midi messages, return a histogram of note probabilities.
        """
        histogram = np.zeros(NUM_NOTES)
        for msg in msglist:
            if not msg.is_meta:
                if msg.type == 'note_on':
                    note_number = self.midi_to_note(msg.note)
                    histogram[note_number] = histogram[note_number] + 1
        histogram = self.normalize(histogram)
        return histogram

    def train_detector(self, midifile, label, model_filename='knn_model.pkl'):
        """
        Given an input midi file, returns a
        12-element vector of values corresponding to the
        probability of each 12 notes from C, C#, D, ... A#, B.

        IMPORTANT:
        The resulting data vector and its label are shifted to all 12
        possible transpositions of that SAME MODE, so the detector is
        trained with 12 new data-label pairs for each midi file.
        """
        if label not in NAME2NOTE:
            print "Invalid note label"
            return

        # Original key
        data_vec = self.file2histogram(midifile)
        self.plot_histogram(data_vec)
        self.data = data_vec
        self.labels = [label]
        # Add 11 other possible keys
        for interval in range(1, NUM_NOTES): # 1-11
            # Musical-tranpose label
            new_note = (NAME2NOTE[label] + interval) % NUM_NOTES
            new_label = NOTE2NAME[new_note]
            # Musical-tranpose (shift/cycle) histogram
            new_data_vec = [data_vec[(i-interval) % NUM_NOTES] for i in range(len(data_vec))]
            self.plot_histogram(new_data_vec)
            # Append to data matrix
            self.data = np.vstack((self.data, new_data_vec))
            self.labels.append(new_label)
        # Fit data using nearest neighbors
        self.knn_model = sklearn.neighbors.NearestNeighbors(n_neighbors=1).fit(self.data)
        # Save model
        sklearn.externals.joblib.dump((self.knn_model, self.labels), model_filename)
        print "Training complete. Model saved in " + model_filename
        return

    def predict(self, msglist):
        """
        Predicts the class/key of a list of messages using nearest-neighbors search
        """
        histogram = self.get_histogram(msglist)
        histogram = histogram.reshape(1, -1) # Comment out this line for multiple samples
        distances, indices = self.knn_model.kneighbors(histogram)
        predicted_labels = self.labels[indices[0][0]]
        return predicted_labels

    def plot_histogram(self, hist):
        """
        Actually plotting a bar plot
        """
        # plt.ion() # Turn on interactive mode
        x = np.arange(len(hist))
        plt.bar(x, height=hist)
        plt.xticks(x+.5, NOTE_NAMES)
        plt.show()

if __name__ == "__main__":
    key_detector = KeyDetector()
    if sys.argv[1] == 'train' and len(sys.argv) == 4:
        key_detector.train_detector(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'predict' and len(sys.argv) == 3:
        # Load previously trained model
        key_detector.load_model()
        # Convert midifile to histogram then predict
        data_vec = key_detector.file2histogram(sys.argv[2])
        print key_detector.predict(data_vec)
    else:
        print "Invalid arguments / Incorrect number of arguments."
        print key_detector.info()