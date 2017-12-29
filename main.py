import mido

if __name__ == "__main__":
    # Look for input ports
    # print mido.get_input_names()
    # print mido.get_output_names()
    # IN_PORT = mido.get_input_names()[0]
    # OUT_PORT = mido.get_output_names()[0]
    with mido.open_input('mido_in', virtual=True) as inport, \
        mido.open_output('mido_out', virtual=True, autoreset=True) as outport:
        # while (1):
        for msg in inport: # Loops until port is closed
            outport.send(msg)