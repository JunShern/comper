import time
import mido

if __name__ == "__main__":
    with mido.open_output('midi_read_out', virtual=True, autoreset=True) as outport:
        for msg in mido.MidiFile('midi_files/abba-winner_takes_it_all.mid'):
            time.sleep(msg.time)
            if not msg.is_meta:
                outport.send(msg)