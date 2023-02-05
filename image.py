# %%
import os
from os.path import join
import numpy as np
import cv2
import matplotlib.pyplot as plt
from side_extractor import process_piece, plot_side_images
from functools import partial
import traceback

# %%
filenames = os.listdir('my_images')
filenames.sort()

# %%
def plot_grid(size, out_dict, *image_keys):
    h, w = size
    for idx, img_key in enumerate(image_keys, start=1):
        plt.subplot(h * 100 + w * 10 + idx)
        if img_key[0] == '_':
            plt.imshow(out_dict[img_key[1:]], cmap='gray')
        else:
            plt.imshow(out_dict[img_key])

# %%
#label_tuples = [('A', 74), ('B', 43), ('C', 19), ('D', 72), ('E', 11)]
label_tuples = [('A', 1)]

def create_label(label_tuple):
    letter, max_num = label_tuple
    for i in range(1, max_num + 1):
        label = letter + str(i) if i >= 10 else letter + '0' + str(i)
        yield label
        
labels = []
for label_tuple in label_tuples:
    for label in create_label(label_tuple):
        labels.append(label)

# %%
postprocess = partial(cv2.blur, ksize=(3, 3))
results = []
error_labels = []
x_offset = 0
y_offset = 0

for filename, label in zip(filenames, labels):
    img = cv2.imread(join('my_images', filename))
    img = img[0:2821, 0:3985]
    imge_size = img.shape
    print(f"imge_size = {imge_size}")
    x_offset = int(imge_size[1]/7)
    y_offset = int(imge_size[0]/5)
    print(f"x_offset = {x_offset} y_offset = {y_offset}")
    for i in range(0, 7): 
        for j in range(0, 5):
            start_point_x = 30 + x_offset * i
            start_point_y = 30 + y_offset * j
            start_point = (start_point_x, start_point_y)
            #start_point = (10, 10)
            window_name = 'Image_'+str(i+1) +"_"+ str(j+1)
            
            end_point_x = start_point[0] + 510 
            end_point_y = start_point[1] + 510
            end_point = (end_point_x, end_point_y)
            #print(f"i = {i} j = {j} start_point={start_point}, end_point={end_point}")
            image = img[start_point_x:end_point_x, start_point_y:end_point_y]
            image = cv2.bitwise_not(image)
            # Blue color in BGR
            color = (255, 0, 0)
            
            # Line thickness of 2 px
            thickness = 2
             
            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            img = cv2.rectangle(img, start_point, end_point, color, thickness)
            #cv2.imshow(window_name, image) 
            out_path = join('output_folder', "out_filename.jpg")
        
            cv2.imwrite(out_path, img)
            cv2.waitKey(0)

            