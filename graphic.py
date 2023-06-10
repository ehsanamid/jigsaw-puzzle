import os
from os.path import join
import tkinter as tk
from PIL import ImageTk
import pandas as pd


# Create a list of images
images = []



df_sides = pd.read_csv('sides.csv')
for index, row in df_sides.iterrows():
    piece_file_name = join("threshold", row['Piece']+".png")
    
    images.append(piece_file_name)


# Create the main window
window = tk.Tk()

# Create a label to display the image
label = tk.Label(window)

# Create a function to load the next image
def next_image():
    global current_image
    current_image = (current_image + 1) % len(images)
    image = ImageTk.PhotoImage(Image.open(images[current_image]))
    label.configure(image=image)

# Create a function to load the previous image
def previous_image():
    global current_image
    current_image = (current_image - 1) % len(images)
    image = ImageTk.PhotoImage(Image.open(images[current_image]))
    label.configure(image=image)

# Create a left arrow button
left_arrow = tk.Button(window, text="Left", command=previous_image)

# Create a right arrow button
right_arrow = tk.Button(window, text="Right", command=next_image)

# Pack the widgets
label.pack()
left_arrow.pack()
right_arrow.pack()

# Set the current image
current_image = 0

# Start the main loop
window.mainloop()