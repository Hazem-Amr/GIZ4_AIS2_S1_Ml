import tkinter as tk
import numpy as np



class salaryPredictionApp:
    def __init__(self , root):
        self.root = root
        self.root.title("DEPI Diploma for AI")
        self.root.geometry("500x400")
        self.create_widgets()

    def create_widgets(self):
        header = tk.Label((self.root), text="DEPI Machine Learning Diploma",
                                        bg = 'blue', fg="white", font=('Arial' , 28 , "bold"))

        header.pack(fill = tk.X)












if __name__ == "__main__":
    root = tk.Tk()
    app = salaryPredictionApp(root)
    root.mainloop()