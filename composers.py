import time
import random
import collections
import itertools
import mido
import mido.frozen
import key_detector
import sklearn.externals
import numpy as np
import unit_predictor
import pianoroll_utils

MEMORY_LENGTH = 10000
COMP_CHANNEL = 10 # 0-15, +1 to get the MIDI channel number (1-16) seen by the user
TIME_PRECISION = 1
KEY_CHANGE_MEMORY_LEN = 70

#------------------------------------------------------------- GENERAL PURPOSE FUNCTIONS

def transpose_message(msg, original_key, target_key):
    if hasattr(msg, 'note'): # Only transpose NOTE_ON/NOTE_OFF
        interval = key_detector.NAME2NOTE[target_key] - key_detector.NAME2NOTE[original_key]
        msg = msg.copy(note=msg.note+interval)
    return msg

#-------------------------------------------------------------

class Composer(object):
    """
    Base class for possible composer types, implements:
    - causal memory states of music
    - callback function for new notes
    - empty comping function
    """
    def __init__(self):
        self.player_messages = collections.deque(maxlen=MEMORY_LENGTH)
        self.gen_messages = collections.deque(maxlen=MEMORY_LENGTH)
        self.active_notes = []
        self.previous_event_time = 0
        self.current_key = 'C'
        self.key_detector = key_detector.KeyDetector()
        self.key_detector.load_model()

    def get_deltatime(self):
        """
        Delta time gives the time since the previous MIDI event
        """
        deltatime = 0 # Arbitrarily set time for first note
        timenow = time.time()
        if self.previous_event_time:
            deltatime = timenow - self.previous_event_time
        self.previous_event_time = timenow
        return deltatime

    def register_player_note(self, msg, precision=None):
        """
        Registers the deltatime of a note_on/note_off message,
        and keeps track of all messages
        """
        if hasattr(msg, 'note'): # Only register NOTE_ON or NOTE_OFF messages
            # Set time attribute
            delta_time = msg.time
            if not delta_time: # Set time based on live playing
                delta_time = self.get_deltatime()
            if precision: # Quantize time
                delta_time = round(delta_time, precision)
            msg = msg.copy(time=delta_time, channel=COMP_CHANNEL)

            # IMPORTANT - need to freeze message to make them hashable
            msg = mido.frozen.freeze_message(msg)
            self.add_to_player_memory(msg)
            self.detect_key()

    def detect_key(self):
        # Detect key from KEY_CHANGE_MEMORY_LEN latest messages
        if self.player_messages:
            start_index = max(0, len(self.player_messages)-KEY_CHANGE_MEMORY_LEN)
            latest_messages = list(itertools.islice(self.player_messages, start_index, None))
            self.current_key = self.key_detector.predict_from_msglist(latest_messages)

    def add_to_player_memory(self, msg):
        """
        Stores note_on/note_off messages in player memory,
        and updates the currently active notes
        """
        if msg.type == "note_on":
            if not msg.note in self.active_notes:
                self.active_notes.append(msg.note)
            self.player_messages.append(msg)
        elif msg.type == "note_off":
            if msg.note in self.active_notes:
                self.active_notes.remove(msg.note)
            self.player_messages.append(msg)
        if len(self.player_messages) == self.player_messages.maxlen:
            print "Maximum memory reached"

    def add_to_gen_memory(self, msg):
        """
        Stores generated messages in gen memory
        """
        self.gen_messages.append(msg)

    def generate_comp(self, _):
        """
        Dummy function - should be overriden in child classes
        """
        time.sleep(0.2)
        return

    def exit(self):
        return

class RandomMemoryDurationless(Composer):
    """
    Simply plays back notes at random from memory of player_messages
    """
    def __init__(self):
        Composer.__init__(self)

    def generate_comp(self, outport):
        if self.player_messages:
            msg = self.player_messages[random.randint(0, len(self.player_messages)-1)]
            msg = msg.copy(channel=COMP_CHANNEL)
            outport.send(msg)
        time.sleep(0.2)

class RandomMemory(Composer):
    """
    Simply plays back messages at random from memory of player_messages
    """
    def __init__(self):
        Composer.__init__(self)

    def generate_comp(self, outport):
        if self.player_messages:
            msg = self.player_messages[random.randint(0, len(self.player_messages)-1)]
            msg = msg.copy(channel=COMP_CHANNEL)
            time.sleep(msg.time)
            outport.send(msg)
        else:
            time.sleep(0.2)

class Arpeggiator(Composer):
    """
    Arpeggiate notes from a currently active notes (eg. held chord)
    """
    def __init__(self):
        Composer.__init__(self)

    def generate_comp(self, outport):
        for note_ in self.active_notes:
            msg = mido.Message('note_on', note=note_, velocity=100, channel=COMP_CHANNEL)
            outport.send(msg)
            time.sleep(0.2)
            msg = mido.Message('note_off', note=note_, velocity=100, channel=COMP_CHANNEL)
            outport.send(msg)

################################################################ MARKOV-GENERATORS

class MarkovBaseClass(Composer):
    """
    Base class for Markov Generator, does not generate notes!
    """
    def __init__(self):
        Composer.__init__(self)
        self.markov_chain = {} # This will be a dictionary of state:[next_states]

    def add_to_chain(self, state, next_state):
        """
        Add to Markov Chain
        """
        if state in self.markov_chain:
            self.markov_chain[state].append(next_state)
        else: # First initialization
            self.markov_chain[state] = [next_state]

    def get_next_state(self, gen_states, player_states):
        """
        Generate from Markov Chain
        """
        if self.markov_chain:
            # Generate new note from Markov Chain (if the state has been registered in the chain)
            if gen_states and (gen_states[-1] in self.markov_chain):
                next_state = random.choice(self.markov_chain[gen_states[-1]])
                return next_state
            # Runs first time, after player has played first notes but before generator has played
            elif player_states and (player_states[-1] in self.markov_chain):
                next_state = random.choice(self.markov_chain[player_states[-1]])
                return next_state
            # else:
            #     print "This only happens when: gen_states are empty, or when generator lands on player's newest note"
        return 0

class MarkovDurationless(MarkovBaseClass):
    """
    Markov generator based only on note frequencies,
    no regard to note durations or other characteristics.
    """
    def __init__(self):
        MarkovBaseClass.__init__(self)

    def register_player_note(self, msg):
        """
        Registers the deltatime of a message,
        and keeps track of all messages
        Also, update Markov Chain.
        """
        MarkovBaseClass.register_player_note(self, msg)
        # Add to Markov Chain
        if msg.type == "note_on" and len(self.player_messages) >= 2:
            self.add_to_chain(self.player_messages[-2].note, self.player_messages[-1].note)

    def generate_comp(self, outport):
        # Prepare state-lists for Markov Generator
        player_states = [msg.note for msg in self.player_messages]
        gen_states = [msg.note for msg in self.gen_messages]
        note_ = self.get_next_state(gen_states, player_states)

        # Send next state
        if note_:
            msg = mido.Message('note_on', note=note_, velocity=100, channel=COMP_CHANNEL)
            self.add_to_gen_memory(msg)
            outport.send(msg)
            time.sleep(0.2)
            msg = mido.Message('note_off', note=note_, velocity=100, channel=COMP_CHANNEL)
            self.add_to_gen_memory(msg)
            outport.send(msg)
        else:
            time.sleep(0.2)

class MarkovQuantizeDuration(MarkovBaseClass):
    """
    Markov generator to produce full MIDI messages (with duration),
    implements quantization of durations (by rounding) to improve note association.
    Training based on player states; Generation based on the generator's states
    """
    def __init__(self):
        MarkovBaseClass.__init__(self)

    def register_player_note(self, msg, precision=None):
        """
        Registers the (quantized) deltatime of a message,
        and keeps track of all messages
        Also, update Markov Chain.
        """
        MarkovBaseClass.register_player_note(self, msg, TIME_PRECISION)
        # Add to Markov Chain
        if len(self.player_messages) >= 2:
            self.add_to_chain(self.player_messages[-2], self.player_messages[-1])

    def generate_comp(self, outport):
        msg = self.get_next_state(self.gen_messages, self.player_messages)
        if msg:
            self.add_to_gen_memory(msg)
            time.sleep(msg.time)
            outport.send(msg)
        else:
            time.sleep(0.2)

class MarkovQuantizeDurationKeyTranspose(MarkovBaseClass):
    """
    Builds on MarkovQuantizeDuration, but also handles automatic transposition
    by detecting the current key the player is in. All learned patterns are
    transposed to the key of C major, then retransposed to suit the current
    key upon generation.
    """
    def __init__(self):
        MarkovBaseClass.__init__(self)

    def register_player_note(self, msg, precision=None):
        """
        Registers the (quantized) deltatime of a message,
        and keeps track of all messages
        Also, update Markov Chain.
        """
        msg = transpose_message(msg, self.current_key, 'C') # Store in key of C
        MarkovBaseClass.register_player_note(self, msg, TIME_PRECISION)
        # Add to Markov Chain
        if len(self.player_messages) >= 2:
            self.add_to_chain(self.player_messages[-2], self.player_messages[-1])

    def generate_comp(self, outport):
        print self.current_key
        msg = self.get_next_state(self.gen_messages, self.player_messages)
        msg = transpose_message(msg, 'C', self.current_key) # Generate in current key
        if msg:
            self.add_to_gen_memory(msg)
            time.sleep(msg.time)
            outport.send(msg)
        else:
            time.sleep(0.2)

class MarkovQuantizeDurationKeyTransposeLongTermMemory(MarkovQuantizeDurationKeyTranspose):
    def __init__(self):
        MarkovQuantizeDurationKeyTranspose.__init__(self)
        INFILE = 'midi_files/all_out_of_love_A#.mid'
        OUTFILE = 'models/markov_1.pkl'
        self.train_markov(INFILE, OUTFILE)
        self.markov_chain = sklearn.externals.joblib.load(OUTFILE)

    def train_markov(self, infile_name, outfile_name):
        """
        Loads a MIDI file (infile_name) into a Markov model,
        then saves the model to a .pkl file (outfile_name)
        """
        midifile = mido.MidiFile(infile_name)
        print "Number of tracks: " + str(len(midifile.tracks))
        for i, track in enumerate(midifile.tracks):
            print "(Track " + str(i) + "): " + str(track.name) + " has " + str(len(track)) + " messages."
        if midifile.type != 2:
            for msg in midifile:
                print self.current_key
                self.register_player_note(msg)
        else:
            print "Asynchronous midi file not supported, exiting..."
        sklearn.externals.joblib.dump(self.markov_chain, outfile_name)
        print self.markov_chain
        print "Training complete. Model saved in " + outfile_name
        return

################################################################ LOOPER

class UnitLooper(Composer):
    """
    Records 4 beats of music and plays it back
    """
    def __init__(self):
        Composer.__init__(self)
        # Prompt user for BPM
        self.beats_per_minute = int(raw_input("Beats per minute: "))
        # Time definitions
        self.beats_per_bar = 4
        self.ticks_per_beat = 24
        self.seconds_per_tick = 60. / self.beats_per_minute / self.ticks_per_beat
        # Prepare for tick-quantized music
        self.num_ticks = self.ticks_per_beat * self.beats_per_bar        
        self.input_events = [[] for _ in range(self.num_ticks)] # Each tick gets a list to store events
        self.comp_events = [[] for _ in range(self.num_ticks)] # Each tick gets a list to store events
        self.current_tick = 0 # This acts as an index for quantized events
        # Prepare pianoroll
        self.num_pitches = 128 
        self.input_pianoroll = np.zeros((self.num_pitches, self.num_ticks))
        self.comp_pianoroll = np.zeros((self.num_pitches, self.num_ticks))
        self.unit_predictor = unit_predictor.UnitPredictor()
        # Alternate between input and response
        self.loopcount = 0

    def generate_comp(self, outport):
        """
        Simultaneously carries out three goals:
        1. Acts as a metronome by playing drum sounds at each beat
            - Drums https://commons.wikimedia.org/wiki/File:GMStandardDrumMap.gif
        2. Plays back recorded user input in a loop
        3. Plays back the generated accompaniment
        """
        # Empty the input
        self.input_pianoroll = np.zeros((self.num_pitches, self.num_ticks))
        self.input_events = [[] for _ in range(self.num_ticks)] # Each tick gets a list to store events

        if self.loopcount % 2 == 0:
            print("Your turn!")
        else:
            print("Comper's turn")
        
        DRUM_CHANNEL = 9
        HI_HAT, BASS, SNARE = (42, 36, 38)
        beat_sounds = [(BASS, HI_HAT), (HI_HAT,), (HI_HAT,), (HI_HAT,)]
        # Loop through every tick in every beat
        for beat in range(self.beats_per_bar):
            # Play drum sounds at each beat
            for sound in beat_sounds[beat]:
                msg = mido.Message('note_on', note=sound, velocity=100, time=0.)
                msg = msg.copy(channel=DRUM_CHANNEL)
                outport.send(msg)
            # Play recorded messages and wait at each tick
            for tick in range(self.ticks_per_beat):
                self.current_tick = beat*self.ticks_per_beat + tick
                if self.loopcount % 2 == 0:
                    for msg in self.input_events[self.current_tick]:
                        outport.send(msg.copy(channel=COMP_CHANNEL, time=0))
                else:
                    for msg in self.comp_events[self.current_tick]:
                        outport.send(msg.copy(channel=COMP_CHANNEL, time=0))
                time.sleep(self.seconds_per_tick)

        # Predict comp events for the next unit
        if np.sum(self.input_pianoroll) > 0:
            self.comp_pianoroll = self.unit_predictor.get_comp_pianoroll(self.input_pianoroll)
            # np.save('recorded_pianoroll.npy', self.input_pianoroll)
            # np.save('generated_pianoroll.npy', self.comp_pianoroll)
            self.comp_events = pianoroll_utils.pianoroll_2_events(self.comp_pianoroll)
        else:
            self.comp_events = [[] for _ in range(self.num_ticks)] # Each tick gets a list to store events
        self.loopcount += 1

    def register_player_note(self, msg, precision=None):
        """
        Registers the deltatime of a note_on/note_off message,
        and keeps track of all messages
        """
        if hasattr(msg, 'note'): # Only register NOTE_ON or NOTE_OFF messages
            # Set time attribute
            delta_time = msg.time
            if not delta_time: # Set time based on live playing
                delta_time = self.get_deltatime()
            if precision: # Quantize time
                delta_time = round(delta_time, precision)
            msg = msg.copy(time=delta_time, channel=COMP_CHANNEL)
            # Store as an event for current tick
            self.input_events[self.current_tick].append(msg)
            # Write to pianoroll
            if msg.type == "note_on":
                self.input_pianoroll[msg.note, self.current_tick:] = msg.velocity
            elif msg.type == "note_off":
                self.input_pianoroll[msg.note, self.current_tick:] = 0

class UnitSelector(UnitLooper):
    def __init__(self):
        UnitLooper.__init__(self)
        self.unit_predictor = unit_predictor.UnitSelector()

class UnitSelectorV2(UnitLooper):
    def __init__(self):
        UnitLooper.__init__(self)
        self.unit_predictor = unit_predictor.UnitSelectorV2()

class UnitAutoencoder(UnitLooper):
    def __init__(self):
        UnitLooper.__init__(self)
        self.unit_predictor = unit_predictor.UnitAutoencoder()

class UnitVariationalAutoencoder(UnitLooper):
    def __init__(self):
        UnitLooper.__init__(self)
        self.unit_predictor = unit_predictor.UnitVariationalAutoencoder()


################################################################ ACCOMPANIMENT
class UnitAccompanier(UnitLooper):
    """
    Predicts a next-state accompaniment unit based on 
    previous input and accompaniment units
    """
    def __init__(self):
        UnitLooper.__init__(self)
        self.unit_predictor = unit_predictor.UnitAccompanier()

    def generate_comp(self, outport):
        """
        Simultaneously carries out two goals:
        1. Acts as a metronome by playing drum sounds at each beat
            - Drums https://commons.wikimedia.org/wiki/File:GMStandardDrumMap.gif
        2. Plays back the generated accompaniment
        """
        # Empty the input
        self.input_pianoroll = np.zeros((self.num_pitches, self.num_ticks))
        self.input_events = [[] for _ in range(self.num_ticks)] # Each tick gets a list to store events

        DRUM_CHANNEL = 9
        HI_HAT, BASS, SNARE = (42, 36, 38)
        beat_sounds = [(BASS, HI_HAT), (HI_HAT,), (HI_HAT,), (HI_HAT,)]
        # Loop through every tick in every beat
        for beat in range(self.beats_per_bar):
            # Play drum sounds at each beat
            for sound in beat_sounds[beat]:
                msg = mido.Message('note_on', note=sound, velocity=100, time=0.)
                msg = msg.copy(channel=DRUM_CHANNEL)
                outport.send(msg)
            # Play recorded messages and wait at each tick
            for tick in range(self.ticks_per_beat):
                self.current_tick = beat*self.ticks_per_beat + tick
                for msg in self.comp_events[self.current_tick]:
                    outport.send(msg.copy(channel=COMP_CHANNEL, time=0))
                time.sleep(self.seconds_per_tick)

        # Predict comp events for the next unit
        self.comp_pianoroll = self.unit_predictor.get_comp_pianoroll(self.input_pianoroll)
        self.comp_events = pianoroll_utils.pianoroll_2_events(self.comp_pianoroll)