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

for filename, label in zip(filenames, labels):
    img = cv2.imread(join('my_images', filename))
    img = img[10:434, 10:445]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply canny edge detection
    plt.imshow(gray, cmap='gray')
    plt.show()
    edges = cv2.Canny(gray, 1, 1)
    plt.imshow(edges, cmap='gray')
    plt.show()
    out_dict = process_piece(img, after_segmentation_func=postprocess, scale_factor=0.4, 
                             harris_block_size=5, harris_ksize=5,
                             corner_score_threshold=0.2, corner_minmax_threshold=100)
    
    plt.figure(figsize=(6, 6))
    plt.title("{0} - {1}".format(filename, label))
    plt.imshow(out_dict['extracted'], cmap='gray')
    plt.scatter(out_dict['xy'][:, 0], out_dict['xy'][:, 1], color='red')
    #plt.colorbar()
    plt.show()
    
    if 'error' in out_dict:
        print( label, ':', out_dict['error'])
        error_labels.append(label)
        traceback.print_exc()
        continue
        
    else:
        
        plt.figure(figsize=(6, 6))
        # plt.title("{0} - {1}".format(filename, label))
        plt.imshow(out_dict['class_image'])
        #plot_grid((3, 3), out_dict, '_segmented', '_extracted', '_edges', 'class_image')
        # plt.show()

        # plot_side_images(out_dict['side_images'], out_dict['inout'])

        results.append({'side_images': out_dict['side_images'], 'inout': out_dict['inout']})


# %%
to_ignore = ['D70']
for el in error_labels:
    labels.remove(el)

for label, result in zip(labels, results):
    
    if label in to_ignore:
        continue
    
    for i, (side_image, io) in enumerate(zip(result['side_images'], result['inout']), start=1):
        
        out_io = 'int' if io == 'in' else 'out'
        side_image = side_image * 255
        
        
        out_filename = "{0}_{1}_{2}.jpg".format(label, i, out_io)
        out_path = join('output_folder', out_filename)
        
        cv2.imwrite(out_path, side_image)


