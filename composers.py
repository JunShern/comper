import time
import random
import collections
import mido
import mido.frozen

MEMORY_LENGTH = 5000
COMP_CHANNEL = 10 # 0-15, +1 to get the MIDI channel number (1-16) seen by the user
PRECISION = 1

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

    def get_deltatime(self, precision=None):
        """
        Delta time gives the time since the previous MIDI event
        """
        deltatime = 0 # Arbitrarily set time for first note
        timenow = time.time()
        if self.previous_event_time:
            deltatime = timenow - self.previous_event_time
        self.previous_event_time = timenow

        if not precision:
            return deltatime
        else:
            # Quantize time
            return round(deltatime, precision)

    def register_player_note(self, msg, precision=None):
        """
        Registers the deltatime of a message,
        and keeps track of all messages
        """
        delta_time = self.get_deltatime(precision)
        msg = msg.copy(time=delta_time, channel=COMP_CHANNEL)
        # IMPORTANT - need to freeze message to make them hashable
        msg = mido.frozen.freeze_message(msg)
        self.add_to_player_memory(msg)

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
        MarkovBaseClass.register_player_note(self, msg, PRECISION)
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
