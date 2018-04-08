import Tkinter

window = Tkinter.Tk()
window.title("Comper")
window.geometry("400x200")

def doAdd():
    numA = int(inputA.get())
    numB = int(inputB.get())
    dispText = str(numA)+" + "+str(numB)+" = "+str(numA+numB)
    ansLabel.configure(text=dispText)

def doSubtract():
    numA = int(inputA.get())
    numB = int(inputB.get())
    dispText = str(numA)+" - "+str(numB)+" = "+str(numA-numB)
    ansLabel.configure(text=dispText)

def doMultiply():
    numA = int(inputA.get())
    numB = int(inputB.get())
    dispText = str(numA)+" * "+str(numB)+" = "+str(numA*numB)
    ansLabel.configure(text=dispText)

def doDivide():
    numA = int(inputA.get())
    numB = int(inputB.get())
    dispText = str(numA)+" / "+str(numB)+" = "+str(numA/numB)
    ansLabel.configure(text=dispText)

# Greeting / Instructions
lbl = Tkinter.Label(window, text="Welcome to my calculator!")
lbl.pack()
# Create text inputs
inputA = Tkinter.Entry(window)
inputA.pack()
inputB = Tkinter.Entry(window)
inputB.pack()
# Create buttons for operators
addButton = Tkinter.Button(window, text="+", command=doAdd)
addButton.pack()
subtractButton = Tkinter.Button(window, text="-", command=doSubtract)
subtractButton.pack()
multiplyButton = Tkinter.Button(window, text="*", command=doMultiply)
multiplyButton.pack()
divideButton = Tkinter.Button(window, text="/", command=doDivide)
divideButton.pack()
# Answer
ansLabel = Tkinter.Label(window, text="Answer goes here")
ansLabel.pack()
# Program
window.mainloop()