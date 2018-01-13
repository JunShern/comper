"""
Key detector based on bag-of-words histogram detection
"""
import sys
import time
import mido
import numpy as np

NOTE_NAME = {0:"C", 1:"C#", 2:"D", 3:"D#", 4:"E", 5:"F", \
    6:"F#", 7:"G", 8:"G#", 9:"A", 10:"A#", 11:"B"}

class KeyDetector(object):
    def __init__(self):
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


    def midi_to_note(self, note_number):
        normalized_note_number = note_number % 12
        return normalized_note_number

    def normalize(self, hist):
        largest_value = max(hist)
        norm_hist = hist / largest_value
        return norm_hist

    def get_histogram(self, midifile):
        histogram = np.zeros(len(NOTE_NAME))
        for msg in mido.MidiFile(midifile):
            if not msg.is_meta:
                if msg.type == 'note_on':
                    note_number = self.midi_to_note(msg.note)
                    histogram[note_number] = histogram[note_number] + 1
        histogram = self.normalize(histogram)
        return histogram

    def train_detector(self, midifile, note_label):
        """
        Given an input midi file, returns a
        12-element list of values corresponding to the
        probability of each 12 notes from C, C#, D, ... A#, B
        """
        data_vec = self.get_histogram(midifile)
        return data_vec

    def predict(self, midifile):
        return "C"

if __name__ == "__main__":
    key_detector = KeyDetector()
    if sys.argv[1] == 'train' and len(sys.argv) == 4:
        print key_detector.train_detector(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'predict' and len(sys.argv) == 3:
        print key_detector.predict(sys.argv[2])
    else:
        print "Incorrect number of arguments."
        print key_detector.info()