"""
Key detector based on bag-of-words histogram detection
"""
import sys
import time
import mido
import operator

NOTE_NAME = {0:"C", 1:"C#", 2:"D", 3:"D#", 4:"E", 5:"F", \
    6:"F#", 7:"G", 8:"G#", 9:"A", 10:"A#", 11:"B"}

class KeyDetector(object):
    def __init__(self):
        self.histogram = [0]*len(NOTE_NAME)

    def midi_to_note(self, note_number):
        normalized_note_number = note_number % 12
        return normalized_note_number

    def normalize(self, hist):
        largest_value = max(hist)
        norm_hist = [0] * len(hist)
        for i, _ in enumerate(hist):
            norm_hist[i] = hist[i] / float(largest_value)
        return norm_hist

    def train_detector(self, midifile):
        """
        Given an input midi file, returns a
        12-element list of values corresponding to the
        probability of each 12 notes from C, C#, D, ... A#, B
        """
        for msg in mido.MidiFile(midifile):
            if not msg.is_meta:
                if msg.type == 'note_on':
                    keyname = self.midi_to_note(msg.note)
                    self.histogram[keyname] = self.histogram[keyname] + 1
        return self.normalize(self.histogram)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        key_detector = KeyDetector()
        hist = key_detector.train_detector(sys.argv[1])
        print hist
    else:
        print "Incorrect number of arguments."