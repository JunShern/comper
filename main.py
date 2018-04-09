import mido
import composers
import tkinter

def callback():
    global done 
    done = True
    print "The end!"

def all_subclasses(cls):
    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in all_subclasses(s)]

# GUI
tk = tkinter.Tk()
btn = tkinter.Button(tk, text="Quit", command=callback)
btn.pack()

done = False # Composer loop
if __name__ == "__main__":
    # User select Composer mode
    comp_classes = all_subclasses(composers.Composer)
    print "Please select a Composer mode: "
    for i, comp_class in enumerate(comp_classes):
        print str(i) + ": " + comp_class.__name__
    comp_index = int(raw_input("Index number: "))

    with mido.open_output('comper_out', virtual=True, autoreset=True) as outport, \
        mido.open_input('comper_in', virtual=True) as inport:
        # Instantiate composer
        comp = comp_classes[comp_index]()
        # Install callback function for input
        inport.callback = comp.register_player_note
        # Run comping
        while not done:
            tk.update_idletasks()
            tk.update()
            comp.generate_comp(outport)

