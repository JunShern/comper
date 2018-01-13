import time
import mido
import sys

if __name__ == "__main__":
    if len(sys.argv) == 2:
        with mido.open_output('midi_read_out', virtual=True, autoreset=True) as outport:
            midifile = mido.MidiFile(sys.argv[1])
            print "Number of tracks: " + str(len(midifile.tracks))
            for i, track in enumerate(midifile.tracks):
                print "(Track " + str(i) + "): " + str(track.name) + " has " + str(len(track)) + " messages."
            if midifile.type != 2:
                for msg in midifile:
                    print msg
                    time.sleep(msg.time)
                    if not msg.is_meta:
                        outport.send(msg)
            else:
                print "Asynchronous midi file not supported, exiting..."
    else:
        print "Incorrect number of arguments."