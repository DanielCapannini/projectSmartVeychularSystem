import cv2
import os
import tkinter as tk

root = tk.Tk()
immagine_tk = tk.PhotoImage(file='img/parcheggio.jpg')
tk.Label(root, image=immagine_tk).pack()
root.mainloop()