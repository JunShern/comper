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
        self.histogram = {}
        for keyname in NOTE_NAME.values():
            self.histogram[keyname] = 0

    def midi_to_note(self, note_number):
        normalized_note_number = note_number % 12
        return NOTE_NAME[normalized_note_number]

    def normalize(self, hist):
        largest_value = max(hist.values())
        for key in hist:
            hist[key] = hist[key] / float(largest_value)
        return hist

    def train_detector(self, midifile):
        for msg in mido.MidiFile(midifile):
            if not msg.is_meta:
                if msg.type == 'note_on':
                    keyname = self.midi_to_note(msg.note)
                    self.histogram[keyname] = self.histogram[keyname] + 1
        return self.histogram

if __name__ == "__main__":
    if len(sys.argv) == 2:
        key_detector = KeyDetector()
        hist = key_detector.train_detector(sys.argv[1])
        print hist
        hist = key_detector.normalize(hist)
        print hist
    else:
        print "Incorrect number of arguments."