
import os
from os.path import join
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from side_extractor import process_piece1
from ChatGPT import compare_images
from functools import partial
import traceback
import math
from PIL import Image



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
        out_path = join('sides', out_filename)
        cv2.imwrite(out_path, side_image)


def save_peice(image,out_dict):        
    out_filename = out_dict['name'] + ".jpg"
    out_path = join('pieces', out_filename)
    cv2.imwrite(out_path, image)





def find_similarity(big_image, small_image):
    try:
        similarity = []
        for i in range(4):
            result = cv2.matchTemplate(big_image, small_image, cv2.TM_CCOEFF_NORMED)
            # rotate small_image 90 degrees
            small_image = cv2.rotate(small_image, cv2.ROTATE_90_CLOCKWISE)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            similarity.append(max_val)
        
        # return the best match
        return max(similarity)
    except Exception as e:
        print(str(e))

def find_the_best_match(df_pieces: pd.DataFrame):
    # Example usage
    try:
        for side_file_name in df_pieces['piece']:
            with open(side_file_name+"_side_similarity.csv", 'w') as file:
                
                temp_string = "side1,similarity1,side2,similarity2,side3,similarity3,side4,similarity4\n"
                file.write(temp_string)
                for piece_file_name in df_pieces['piece']:
                    if(side_file_name == piece_file_name):
                        continue
                    big_image = cv2.imread(join('contours', piece_file_name + ".jpg"))  
                    side_similarity = []
                    temp_string = ""
                    for i in range(1,5):
                        small_image = cv2.imread(join('sides', side_file_name + "_" + str(i) + ".jpg"))
                        similarity = find_similarity(big_image, small_image)
                        side_similarity.append(similarity)
                        temp_string += f"{piece_file_name},{similarity},"
                    temp_string = temp_string[:-1]+"\n"
                    file.write(temp_string)
    except Exception as e:
        print(str(e))
            
        
# function to read image from camer folder and copy the piece in the pieces folder
def read_camera_image(piece_file_name: str,input_filename: str,\
                        camera_folder: str,piece_folder: str,df_pieces: pd.DataFrame):

    img = cv2.imread(join(camera_folder, input_filename))
    img = img[1100:1700,1400:2000]
    img = cv2.GaussianBlur(img,(3,3),0)
    ret, thr = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)  
    cv2.imwrite(join(piece_folder, piece_file_name), img)
    cv2.imwrite(join(piece_folder+"_threshold", piece_file_name), thr)
    piece_name = os.path.splitext(piece_file_name)[0]
    new_row = pd.DataFrame({'piece':piece_name, \
        'status':'new', \
        'X1':-1, \
        'Y1':-1, \
        'IO1':-1, \
        'X2':-1, \
        'Y2':-1, \
        'IO2':-1, \
        'X3':-1, \
        'Y3':-1, \
        'IO3':-1, \
        'X4':-1, \
        'Y4':-1, \
        'IO4':-1, \
            },index=[0])
    df_pieces = pd.concat([new_row,df_pieces.loc[:]]).reset_index(drop=True)
    return df_pieces


# function to read image from camer folder and copy the piece in the pieces folder
def read_camera_images(page_number: int,camera_folder: str,piece_folder: str,df_pieces):
    filenames = os.listdir(camera_folder)
    filenames.sort()
    i = 1
    j = 1
    already_processed = True
    for filename in filenames:
        piece_file_name = f"Page_{page_number:04d}_{i}_{j}"
        if (df_pieces['piece'].eq(piece_file_name)).any():
            continue
        j = j + 1
        if(j > 7):
            j =1
            i = i+1
        df_pieces = read_camera_image(piece_file_name+".jpg",filename,\
                        camera_folder,piece_folder,df_pieces)
    return df_pieces


def find_corner(piece_name: str,piece_folder: str):
    img = cv2.imread(join(piece_folder, piece_name+".jpg"))

    # xy = (100,200)
    # color2 = (0,0,255)
    # cv2.circle(img,xy,2,color=color2,thickness=3)
    # cv2.imshow("piece_name",img)
    # full_key = cv2.waitKeyEx(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ''' gray = np.float32(gray)
    harris_corners = cv2.cornerHarris(gray, 9, 9, 0.06)
    kernel = np.ones((7, 7), np.uint8)
    harris_corners = cv2.dilate(harris_corners, kernel, iterations=2)
    thr[harris_corners > 0.01 * harris_corners.max()] = [255, 127, 127]'''

    corners = cv2.goodFeaturesToTrack(gray, 100, 0.2, 10)
    x = 0
    y = 0
    df = pd.DataFrame()
    if corners is not None:
        for corner in corners:
            x1, y1 = corner[0]
            x += int(x1)
            y += int(y1)
        x = x/len(corners)
        y = y/len(corners)
        for i, corner in enumerate(corners):
            m = (corner[0][1] - y)/(corner[0][0] - x)
            m = math.atan(m)
            m = math.degrees(m)
            m = abs(abs(m) - 45) 
            x1 = int(corner[0][0])
            y1 = int(corner[0][1])
            new_row = pd.DataFrame({'Index': i,'X':x1,'Y':y1,'angle':m },index=[0])
            df = df.append(new_row)
    
        df.sort_values(by='angle', inplace=True)
        print(df)
        ########
        l = len(corners)
        output_list = []
        count = 0
        for index, row in df.iterrows():
            x = int(row['X'])
            y = int(row['Y'])
            xy = (x,y)
            
            color2 = (0,0,255)
            cv2.circle(img,xy,2,color=color2,thickness=3)
            cv2.imshow(piece_name,img)
            full_key = cv2.waitKeyEx(0)
            if full_key == 110:
                continue
            if full_key == 120:
                break
            count += 1
            output_list.append(xy)
            if(count == 4):
                break
            
            
        cv2.destroyWindow(piece_name)
        ###############
        
        # print(df)
    return output_list

# write a function to search in df_pieces in 'piece' column for the piece_name and read the 'status' column value
# if status is 'new' then call find_corner function and update the df_pieces with the new values

def get_corners(piece_folder: str,df_pieces):
    # filenames = os.listdir(piece_folder)
    
    for piece_file_name in df_pieces['piece']:
        # row_num = df_pieces[df_pieces['piece'] == piece_file_name].index 
        value = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'status'].iloc[0]
        # value = df_pieces.loc[row_num,['status']]
        if(value == 'new'):
            corners = find_corner(piece_file_name,piece_folder)
            if(len(corners) == 4):
                df_pieces.loc[df_pieces['piece'] == piece_file_name, 'status'].iloc[0] = 'p'
                df_pieces.loc[df_pieces['piece'] == piece_file_name, 'X1'].iloc[0] = corners[0][0]
                df_pieces.loc[df_pieces['piece'] == piece_file_name, 'Y1'].iloc[0] = corners[0][1]
                df_pieces.loc[df_pieces['piece'] == piece_file_name, 'X2'].iloc[0] = corners[1][0]
                df_pieces.loc[df_pieces['piece'] == piece_file_name, 'Y2'].iloc[0] = corners[1][1]
                df_pieces.loc[df_pieces['piece'] == piece_file_name, 'X3'].iloc[0] = corners[2][0]
                df_pieces.loc[df_pieces['piece'] == piece_file_name, 'Y3'].iloc[0] = corners[2][1]
                df_pieces.loc[df_pieces['piece'] == piece_file_name, 'X4'].iloc[0] = corners[3][0]
                df_pieces.loc[df_pieces['piece'] == piece_file_name, 'Y4'].iloc[0] = corners[3][1]
    return df_pieces

# function to show the image with corners
def show_image_with_corners(piece_folder: str,df_pieces):
    # filenames = os.listdir(piece_folder)
    for piece_file_name in df_pieces['piece']:
        color2 = (0,0,255)
        img = cv2.imread(join(piece_folder, piece_file_name+".jpg"))
        x1 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'X1'].iloc[0]
        y1 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'Y1'].iloc[0] 
        x = int(x1)
        y = int(y1)
        xy = (x,y)
        cv2.circle(img,xy,2,color=color2,thickness=3)
        x1 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'X2'].iloc[0]
        y1 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'Y2'].iloc[0] 
        x = int(x1)
        y = int(y1)
        xy = (x,y)
        cv2.circle(img,xy,2,color=color2,thickness=3)
        x1 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'X3'].iloc[0]
        y1 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'Y3'].iloc[0] 
        x = int(x1)
        y = int(y1)
        xy = (x,y)
        cv2.circle(img,xy,2,color=color2,thickness=3)
        x1 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'X4'].iloc[0]
        y1 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'Y4'].iloc[0]  
        x = int(x1)
        y = int(y1)
        xy = (x,y)
        cv2.circle(img,xy,2,color=color2,thickness=3)
        cv2.imshow(piece_file_name,img)
        full_key = cv2.waitKeyEx(0)
        cv2.destroyWindow(piece_file_name)


def detect_side_images(df_pieces: pd.DataFrame,pieces_folder: str,sides_folder: str):
    postprocess = partial(cv2.blur, ksize=(3, 3))
    for piece_file_name in df_pieces['piece']:
        if(df_pieces.loc[df_pieces['piece'] == piece_file_name, 'status'].iloc[0] == 'p'):
            img = cv2.imread(join(pieces_folder, piece_file_name+".jpg"))
            out_dict = {}
            out_dict['name'] = piece_file_name
            get_side_image(img,out_dict,df_pieces,piece_file_name,sides_folder,postprocess)
            df_pieces.loc[df_pieces['piece'] == piece_file_name, 'status'] = 's'
            
    return df_pieces

# function to detect side image for all pices bbased on the corners writen in df_pieces
def get_side_image(img,out_dict,df_pieces: pd.DataFrame,piece_file_name: str,sides_folder: str,postprocess):
    
    x1 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'X1'].iloc[0]
    y1 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'Y1'].iloc[0]
    x2 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'X2'].iloc[0]
    y2 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'Y2'].iloc[0]
    x3 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'X3'].iloc[0]
    y3 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'Y3'].iloc[0]
    x4 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'X4'].iloc[0]
    y4 = df_pieces.loc[df_pieces['piece'] == piece_file_name, 'Y4'].iloc[0]
    xy_array = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    # xy_array = out_dict['xy']
    # xy = order_corners(xy_array)
    xy = order_points_clockwise(xy_array)
    out_dict['xy'] = xy
    
    gray = process_piece1(img,out_dict=out_dict, after_segmentation_func=postprocess, scale_factor=0.4, 
                             harris_block_size=5, harris_ksize=5,
                             corner_score_threshold=0.2, corner_minmax_threshold=100)
    
    
    
    # extract_edges(out_dict)
    # plt.figure(figsize=(6, 6))
    # plt.title(out_dict['name'])
    # plt.imshow(out_dict['extracted'], cmap='gray')
    # plt.scatter(out_dict['xy'][:, 0], out_dict['xy'][:, 1], color='red')
    # plt.show()


def order_points_clockwise(pts):
    # Initialize the list of ordered points
    ordered_pts = [None] * 4
    
    # Find the center of the points
    center = [sum(pt[0] for pt in pts) // len(pts), sum(pt[1] for pt in pts) // len(pts)]
    
    # Divide the points into two groups: those above the center and those below
    above_center = []
    below_center = []
    for pt in pts:
        if pt[1] < center[1]:
            above_center.append(pt)
        else:
            below_center.append(pt)
    
    # Sort the points in each group by their x-coordinate
    above_center = sorted(above_center, key=lambda pt: pt[0])
    below_center = sorted(below_center, key=lambda pt: pt[0], reverse=True)
    
    # Assign the ordered points to the output list
    ordered_pts[0] = above_center[0]
    ordered_pts[1] = above_center[-1]
    ordered_pts[2] = below_center[0]
    ordered_pts[3] = below_center[-1]
    
    return ordered_pts




def main():
    
    df_pieces = pd.read_csv('pieces.csv')
    df_sides = pd.read_csv('sides.csv')
    df_pieces = read_camera_images(page_number:= 1,camera_folder:='camera',piece_folder:='pieces',df_pieces)
    df_pieces.to_csv('pieces.csv', index=False)
    df_pieces = get_corners('pieces_threshold',df_pieces)
    
    # show_image_with_corners('pieces_threshold',df_pieces)

    df_pieces = detect_side_images(df_pieces,"pieces_threshold","sides")
    df_pieces.to_csv('pieces.csv', index=False)

    # df_pieces.to_csv('pieces.csv', index=False)
    find_the_best_match(df_pieces)

if __name__ == "__main__":
    main()