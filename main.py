import mido
import time
import composers

if __name__ == "__main__":
    with mido.open_output('comper_out', virtual=True, autoreset=True) as outport, \
        mido.open_input('comper_in', virtual=True) as inport:
        # Instantiate composer
        comp = composers.RandomMemory()
        # Install callback function for input
        inport.callback = comp.add_to_player_memory
        # Run comping
        while 1:
            time.sleep(.2)
            comp.generate_comp(outport)
