
import os
from os.path import join
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from side_extractor import process_piece1,process_piece2, plot_side_images
from ChatGPT import compare_images
from functools import partial
import traceback



def plot_grid(size, out_dict, *image_keys):
    h, w = size
    for idx, img_key in enumerate(image_keys, start=1):
        plt.subplot(h * 100 + w * 10 + idx)
        if img_key[0] == '_':
            plt.imshow(out_dict[img_key[1:]], cmap='gray')
        else:
            plt.imshow(out_dict[img_key])

def puzzel_piece(img,out_dict,df_pieces,df_sides,postprocess)-> bool:    
    gray = process_piece1(img,out_dict=out_dict, after_segmentation_func=postprocess, scale_factor=0.4, 
                             harris_block_size=5, harris_ksize=5,
                             corner_score_threshold=0.2, corner_minmax_threshold=100)
    
    color1 = (255, 0, 0)
    color2 = (0, 255, 0)
    # new_xy = np.zeros((2,0))
    str1 = out_dict['name'] + ","
    xy_array = out_dict['xy']
    l = len(out_dict['xy'])
    for xy in out_dict['xy']:
        cv2.circle(img,xy,2,color=color1,thickness=3)
        cv2.imshow(out_dict['name'],img)
    
    full_key = cv2.waitKeyEx(0)    
    
    for j in range(l,0,-1):
        i = j-1
        xy = out_dict['xy'][i]
        str1 += str(xy[0])+","+str(xy[1])+","
        cv2.circle(img,xy,2,color=color2,thickness=3)
        cv2.imshow(out_dict['name'],img)
        full_key = cv2.waitKeyEx(0)
        if full_key == 110:
            xy_array = np.delete(xy_array, i,0)
    cv2.destroyWindow(out_dict['name'])
    # str1 = str1[:-1]
    # str1 += "\n"
    # f.write(str1)
    # print(xy_array)
    out_dict['xy'] = xy_array
    
    process_piece2(out_dict, after_segmentation_func=postprocess, scale_factor=0.4, 
                             harris_block_size=5, harris_ksize=5,
                             corner_score_threshold=0.2, corner_minmax_threshold=100)

    # plt.figure(figsize=(6, 6))
    # plt.title(out_dict['name'])
    # plt.imshow(out_dict['extracted'], cmap='gray')
    
    # plt.scatter(out_dict['xy'][:, 0], out_dict['xy'][:, 1], color='red')
    
    # plt.show()
    return True
    
    

def plot_image(out_dict):
                
    plt.figure(figsize=(6, 6))
    plt.title(out_dict['name'])
    plt.imshow(out_dict['extracted'], cmap='gray')
    # plt.imshow(out_dict['segmented'])
    plt.scatter(out_dict['xy'][:, 0], out_dict['xy'][:, 1], color='red')
    #plt.colorbar()
    plt.show()


def plot_images(results):
    for out_dict in results:
        plot_image(out_dict)




def create_label(label_tuple):
    letter, max_num = label_tuple
    for i in range(1, max_num + 1):
        label = letter + str(i) if i >= 10 else letter + '0' + str(i)
        yield label

def extract_edges(out_dict):
       
    for i, (side_image, io) in enumerate(zip(out_dict['side_images'], out_dict['inout']), start=1):
        
        out_io = 'int' if io == 'in' else 'out'
        side_image = side_image * 255
        
        
        out_filename = "{0}_{1}_{2}.jpg".format(out_dict['name'], i, out_io)
        out_path = join('output_folder', out_filename)
        
        cv2.imwrite(out_path, side_image)


def save_peice(image,out_dict):        
    out_filename = out_dict['name'] + ".jpg"
    out_path = join('pieces', out_filename)
    
    cv2.imwrite(out_path, image)

def update_pieces_dataframes(df_pieces,out_dict):
    new_row = pd.DataFrame({'piece':out_dict['name'], \
        'X1':out_dict['xy'][0][0], \
        'Y1':out_dict['xy'][0][1], \
        'X2':out_dict['xy'][1][0], \
        'Y2':out_dict['xy'][1][1], \
        'X3':out_dict['xy'][2][0], \
        'Y3':out_dict['xy'][2][1], \
        'X4':out_dict['xy'][3][0], \
        'Y4':out_dict['xy'][3][1], \
            },index=[0])
    df_pieces = pd.concat([new_row,df_pieces.loc[:]]).reset_index(drop=True)
    return df_pieces


def update_sides_dataframes(df_sides,out_dict):
    dic = {}
    dic['side'] = out_dict['name']

    for i, (side_image, io) in enumerate(zip(out_dict['side_images'], out_dict['inout']), start=1):
        x = "w" + str(i)
        y = "h" + str(i)
        ino = "io" + str(i)
        
        dic[x] = side_image.shape[1]
        dic[y] = side_image.shape[0]
        out_io = 0 if io == 'in' else 1
        dic[ino] = out_io
    new_row = pd.DataFrame(dic,index=[0])
    df_sides = pd.concat([new_row,df_sides.loc[:]]).reset_index(drop=True)

    return df_sides

def update_dataframes(df_pieces,df_sides,out_dict):
    df_pieces = update_pieces_dataframes(df_pieces,out_dict)
    df_sides = update_sides_dataframes(df_sides,out_dict)
    return df_pieces,df_sides




def find_image(big_image, small_image):
    result = cv2.matchTemplate(big_image, small_image, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= 0.5:
        return max_loc
    else:
        return False



def main():
    
    # Example usage
    big_image = cv2.imread("large_image.jpg")
    small_image = cv2.imread(join('output_folder', "IMG_001_1_1_3_out.jpg"))

    result = find_image(big_image, small_image)
    if result:
        print("Small image found at position: ", result)
    else:
        print("Small image not found in big image")


    filenames = os.listdir('my_images')
    filenames.sort()


    postprocess = partial(cv2.blur, ksize=(3, 3))
    # results = []
    # error_labels = []


    x_offset = 0
    y_offset = 0

    changed = False
    df_pieces = pd.read_csv('pieces.csv')
    df_sides = pd.read_csv('sides.csv')    
    for filename in filenames:
        img = cv2.imread(join('my_images', filename))
        img = img[0:2821, 0:3985]
        imge_size = img.shape
        # print(f"imge_size = {imge_size}")
        x_offset = int(imge_size[1]/7)
        y_offset = int(imge_size[0]/5)
        # print(f"x_offset = {x_offset} y_offset = {y_offset}")
        #for i in range(0, 5): 
        for i in range(0, 1): 
            #for j in range(0, 7):
            for j in range(0, 1):
                window_name = os.path.splitext(filename)[0] + '_' + str(i+1) + "_" + str(j+1)
                # if(window_name in df_pieces['piece'].unique()):
                #     print(f"puzzle {window_name} already exists")
                #     continue
                start_point_x = 30 + x_offset * i
                start_point_y = 30 + y_offset * j
                start_point = (start_point_x, start_point_y)
                #start_point = (10, 10)
                
                
                end_point_x = start_point[0] + 510 
                end_point_y = start_point[1] + 510
                end_point = (end_point_x, end_point_y)
                #print(f"i = {i} j = {j} start_point={start_point}, end_point={end_point}")
                image = img[start_point_x:end_point_x, start_point_y:end_point_y]
                image = cv2.bitwise_not(image)
                out_dict = {}
                out_dict['name'] = window_name
                save_peice(image,out_dict)
                if(puzzel_piece(image,out_dict,df_pieces,df_sides,postprocess)):
                    extract_edges(out_dict)
                    df_pieces,df_sides = update_dataframes(df_pieces,df_sides,out_dict)
                    changed = True
                side_image = out_dict['class_image'] * 255
                # cv2.imshow(window_name, side_image) 
                # cv2.waitKey(0)
                
        
                cv2.imwrite("large_image.jpg", side_image)
                print(f"puzzel piece {window_name} done")
                

    # plot_images()
    # if(changed):
    #     df_pieces.to_csv('pieces.csv', index=False)
    #     df_sides.to_csv('sides.csv', index=False)
    

    # Creating Empty DataFrame and Storing it in variable df
    
    
    piece_filenames = os.listdir('output_folder')
    piece_filenames.sort()
    
    
    for filename in piece_filenames:
        ret = compare_images(filename,piece_filenames,"output_folder")
        df = pd.DataFrame() 
        df['piece'] = piece_filenames
        df["_similarity"] = ret
        name = os.path.splitext(filename)[0]+".csv"
        df.to_csv(name, index=False)


if __name__ == "__main__":
    main()