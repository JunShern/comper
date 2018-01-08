import mido
import composers

if __name__ == "__main__":
    with mido.open_output('comper_out', virtual=True, autoreset=True) as outport, \
        mido.open_input('comper_in', virtual=True) as inport:
        # Instantiate composer
        comp = composers.MarkovQuantizeDuration()
        # Install callback function for input
        inport.callback = comp.register_player_note
        # Run comping
        while 1:
            comp.generate_comp(outport)
