
import os
from os.path import join
import numpy as np
import pandas as pd
import cv2
from piece import Piece
from side_extractor import process_piece1,get_image_geometry
import math
from PIL import Image






# function to return similarity between two lists of points
def match_contour2(big_image, small_image):
    # convert images to grayscale
    big_image = cv2.cvtColor(big_image, cv2.COLOR_BGR2GRAY)
    small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    # find contours in big_image
    _, big_contours, _ = cv2.findContours(big_image, \
                                          cv2.RETR_TREE, \
                                            cv2.CHAIN_APPROX_SIMPLE)
    # find contours in small_image
    _, small_contours, _ = cv2.findContours(small_image, \
                                            cv2.RETR_TREE, \
                                                cv2.CHAIN_APPROX_SIMPLE)
    # find the biggest contour in big_image
    big_contour = max(big_contours, key=cv2.contourArea)
    # find the biggest contour in small_image
    small_contour = max(small_contours, key=cv2.contourArea)
    # match contours
    match = cv2.matchShapes(big_contour, small_contour, 1, 0.0)
    return match



def find_similarity(big_image,piece_io:list, small_image,side_io: int):
    try:
        similarity = []
        for i in range(4):
            if(piece_io[i] == side_io):
                similarity.append(0)
            else:
                result = cv2.matchTemplate(big_image, small_image, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                similarity.append(max_val)
            # rotate small_image 90 degrees
            small_image = cv2.rotate(small_image, cv2.ROTATE_90_CLOCKWISE)
        # return the best match
        return max(similarity), similarity.index(max(similarity))+1
    except Exception as e:
        print(str(e))

def find_similarity1(big_image,piece_io:list, small_image,side_io: int):
    try:
        similarity = []
        for i in range(4):
            if(piece_io[i] == side_io):
                similarity.append(100000)
            else:
                result = match_contour(big_image=big_image, small_image= small_image)
                similarity.append(result)
            # rotate small_image 90 degrees
            small_image = cv2.rotate(small_image, cv2.ROTATE_90_CLOCKWISE)
        #return the best match and index of the best match
        return min(similarity), similarity.index(min(similarity))+1
    except Exception as e:
        print(str(e))



'''
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray1, 128, 255, cv2.THRESH_BINARY_INV) 

        xy = out_dict['xy']
        edged = cv2.Canny(gray,30,200)
    

        # get the countours of the piece
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        
        # find the closes point in the contour to a point
        contour = max(contours, key=cv2.contourArea)
'''

def match_contour(small_image, big_image):
    # Convert images to grayscale
    small_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    big_gray = cv2.cvtColor(big_image, cv2.COLOR_BGR2GRAY)

    # Find contours in both images
    small_contours, _ = cv2.findContours(small_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_contours, _ = cv2.findContours(big_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_cnt = max(big_contours, key=cv2.contourArea)
    small_cnt = max(small_contours, key=cv2.contourArea)
    best_match = 10000000
    # Loop over the contours in the small image
    
    best_match = cv2.matchShapes(small_cnt, big_cnt, cv2.CONTOURS_MATCH_I1, 0)

    

    # If no match is found, return None
    return best_match


def find_the_best_match():
    # Example usage
    try:

        filenames = os.listdir('sides')
        filenames.sort()
        for side_name in df['piece']:
            # remove "_in" or "_out" from side_name
            if(side_name.endswith("_in")):
                s_name = side_name[:-3]
                s_orientation = 0
            elif(side_name.endswith("_out")):
                s_name = side_name[:-4]
                s_orientation = 1
            with open(side_name+"_side_similarity.csv", 'w') as file:
                
                temp_string = "side1,similarity1,index1,\
                                side2,similarity2,index2,\
                                side3,similarity3,index3,\
                                side4,similarity4,index4\n"
                file.write(temp_string)
                # for piece_name in df['piece']:
                for filename in filenames:
                    piece_name = filename.split("_")[0]
                    # remove "_in" or "_out" from piece_name
                    
                    if(piece_name.endswith("_in")):
                        p_name = piece_name[:-3]
                        p_orientation = 0
                    elif(piece_name.endswith("_out")):
                        p_name = piece_name[:-4]
                        p_orientation = 1

                    if(p_name == s_name) or (s_orientation == p_orientation):
                        continue
                    big_image = cv2.imread(join('contours', piece_name + ".jpg"))  
                    side_similarity = []
                    temp_string = ""
                    
                    small_image = cv2.imread(join('sides', \
                                                    side_name  + ".jpg"))
                    # get IO for the piece from df_pieces
                    side_io = df.loc[df['piece'] == side_name, 'IO'+str(i)].values[0]
                    piece_io = []
                    for j in range(1,5):
                        piece_io.append(\
                            df.loc[df['piece'] == piece_name, 'IO'+str(j)].values[0])
                    similarity,index = find_similarity(\
                        big_image,piece_io, small_image,side_io)
                    side_similarity.append(similarity)
                    temp_string += f"{piece_name},{similarity},{index},"



                    temp_string = temp_string[:-1]+"\n"
                    file.write(temp_string)
    except Exception as e:
        print(str(e))

def find_the_best_matchs():
    # Example usage
    try:

        file_names = os.listdir('sides')
        file_names.sort()
        for file_name in file_names:
            side_name = file_name.split("_")[0]
            # remove "_in" or "_out" from side_name
            if(side_name.endswith("_in")):
                s_name = side_name[:-3]
                s_orientation = 0
            elif(side_name.endswith("_out")):
                s_name = side_name[:-4]
                s_orientation = 1
            with open(side_name+"_side_similarity.csv", 'w') as file:
                
                temp_string = "side1,similarity1,index1,\
                                side2,similarity2,index2,\
                                side3,similarity3,index3,\
                                side4,similarity4,index4\n"
                file.write(temp_string)
                # for piece_name in df['piece']:
                for filename in filenames:
                    piece_name = filename.split("_")[0]
                    # remove "_in" or "_out" from piece_name
                    
                    if(piece_name.endswith("_in")):
                        p_name = piece_name[:-3]
                        p_orientation = 0
                    elif(piece_name.endswith("_out")):
                        p_name = piece_name[:-4]
                        p_orientation = 1

                    if(p_name == s_name) or (s_orientation == p_orientation):
                        continue
                    big_image = cv2.imread(join('contours', piece_name + ".jpg"))  
                    side_similarity = []
                    temp_string = ""
                    
                    small_image = cv2.imread(join('sides', \
                                                    side_name  + ".jpg"))
                    # get IO for the piece from df_pieces
                    side_io = df.loc[df['piece'] == side_name, 'IO'+str(i)].values[0]
                    piece_io = []
                    for j in range(1,5):
                        piece_io.append(\
                            df.loc[df['piece'] == piece_name, 'IO'+str(j)].values[0])
                    similarity,index = find_similarity(\
                        big_image,piece_io, small_image,side_io)
                    side_similarity.append(similarity)
                    temp_string += f"{piece_name},{similarity},{index},"



                    temp_string = temp_string[:-1]+"\n"
                    file.write(temp_string)
    except Exception as e:
        print(str(e))
            

def find_geometries():
    # Example usage
    try:
        list_len = 0
        cols = []
        # return list of csv files in sides folder
        
        file_names = os.listdir('sides')
        file_names.sort()

        for file_name in file_names:
            if(file_name.endswith(".csv")):
                side_name = file_name.split(".")[0]
                # remove "_in" or "_out" from side_name
                # if(side_name.endswith("_in")):
                #     s_name = side_name[:-3]
                #     s_orientation = 0
                # elif(side_name.endswith("_out")):
                #     s_name = side_name[:-4]
                #     s_orientation = 1
                geometry = get_image_geometry(join('sides', file_name))
                geometry.insert(0, side_name)
                if(list_len == 0):
                    list_len = len(geometry)
                    # cols.append('Name')
                    # for i in range(1,list_len):
                    #     cols.append('Col'+str(i))
                    # create a list of column names from 1 to list_len
                    cols = ['Col'+ str(c) for c in range(1, list_len)]
                    cols.insert(0, 'Name')
                    df = pd.DataFrame(columns = cols)
                    df.loc[len(df)] = geometry
                    print(file_name)
                else:
                    df.loc[len(df)] = geometry
                    print(file_name)
        df.to_csv("geometry.csv", index=False)


                    

                # temp_string = temp_string[:-1]+"\n"
                # file.write(temp_string)
    except Exception as e:
        print(str(e))

def transparent():
    # Example usage
    try:
        
        file_names = os.listdir('sides')
        file_names.sort()

        for file_name in file_names:
            
            side_name = file_name.split(".")[0]
           
            # Read the image
            src = cv2.imread(join('sides', file_name), 1)
            
            # Convert image to image gray
            tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            
            # Applying thresholding technique
            _, alpha = cv2.threshold(tmp, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Using cv2.split() to split channels 
            # of coloured image
            b, g, r = cv2.split(src)
            
            # Making list of Red, Green, Blue
            # Channels and alpha
            rgba = [b, g, r, alpha]
            
            # Using cv2.merge() to merge rgba
            # into a coloured/multi-channeled image
            dst = cv2.merge(rgba, 4)
            
            # Writing and saving to a new image
            cv2.imwrite(join('transparent', side_name+".png"), dst)


                    

                # temp_string = temp_string[:-1]+"\n"
                # file.write(temp_string)
    except Exception as e:
        print(str(e))

def transparent1():
    # Example usage
    try:
        
        file_names = os.listdir('sides')
        file_names.sort()

        for file_name in file_names:
            
            side_name = file_name.split(".")[0]
           
            # Read the image
            
            img = Image.open(join('sides', file_name))
            img.save(join('transparent', side_name+".png"))

        file_names = os.listdir('transparent')
        file_names.sort()
        for file_name in file_names:
            
            side_name = file_name.split(".")[0]
           
            # Read the image
            
            img = Image.open(join('transparent', file_name))

            img = img.convert('RGBA')
            datas = img.getdata()

            newData = []
            for item in datas:
                if not(item[0] == 255 and item[1] == 255 and item[2] == 255):
                    # newData.append(item)
                    newData.append((0, 0, 0, 255))
                else:
                    newData.append((255, 255, 255, 0))
            img.putdata(newData)
            img.save(join('transparent', side_name+".png"))

    except Exception as e:
        print(str(e))



# function to read image from camer folder and copy the piece in the pieces folder
""" def read_camera_image(piece_file_name: str,input_filename: str,\
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
 """

# function to read image from camer folder and copy the piece in the pieces folder
def read_camera_images(page_number: int, df: pd.DataFrame):
    filenames = os.listdir("camera")
    filenames.sort()
    i = 1
    j = 1
    
    for filename in filenames:
        piece_file_name = f"Page_{page_number:04d}_{i}_{j}"
        j = j + 1
        if(j > 7):
            j =1
            i = i+1
        if (df['piece'].eq(piece_file_name)).any() and \
            (df.loc[df['piece'] == piece_file_name, 'status'].iloc[0] == 's'):
            continue
        # status of a piece
        # status = df.loc[df['piece'] == piece_file_name, 'status'].iloc[0]
        status = "n"
        piece = Piece(piece_file_name)
        if(piece.read_camera_image(filename,status)):
            new_row = pd.DataFrame({'piece':piece_file_name, \
            'status':'s', \
            'X1':piece.corners[0][0], \
            'Y1':piece.corners[0][1], \
            'IO1':piece.in_out[0], \
            'X2':piece.corners[1][0], \
            'Y2':piece.corners[1][1], \
            'IO2':piece.in_out[1], \
            'X3':piece.corners[2][0], \
            'Y3':piece.corners[2][1], \
            'IO3':piece.in_out[2], \
            'X4':piece.corners[3][0], \
            'Y4':piece.corners[3][1], \
            'IO4':piece.in_out[3] \
                },index=[0])
            df = pd.concat([new_row,df.loc[:]]).reset_index(drop=True)
            print(f"{piece_file_name} added\n")
    df.to_csv('pieces.csv', index=False)
    
        
    # return df_pieces


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

def get_corners(piece_folder: str,df):
    # filenames = os.listdir(piece_folder)
    
    for name in df['piece']:
        # row_num = df_pieces[df_pieces['piece'] == piece_file_name].index 
        value = df.loc[df['piece'] == name, 'status'].iloc[0]
        # value = df_pieces.loc[row_num,['status']]
        if(value == 'new'):
            corners = find_corner(name,piece_folder)
            if(len(corners) == 4):
                df.loc[df['piece'] == name, 'status'].iloc[0] = 'p'
                df.loc[df['piece'] == name, 'X1'].iloc[0] = corners[0][0]
                df.loc[df['piece'] == name, 'Y1'].iloc[0] = corners[0][1]
                df.loc[df['piece'] == name, 'X2'].iloc[0] = corners[1][0]
                df.loc[df['piece'] == name, 'Y2'].iloc[0] = corners[1][1]
                df.loc[df['piece'] == name, 'X3'].iloc[0] = corners[2][0]
                df.loc[df['piece'] == name, 'Y3'].iloc[0] = corners[2][1]
                df.loc[df['piece'] == name, 'X4'].iloc[0] = corners[3][0]
                df.loc[df['piece'] == name, 'Y4'].iloc[0] = corners[3][1]
    return df

# function to show the image with corners
def show_image_with_corners(piece_folder: str,df):
    # filenames = os.listdir(piece_folder)
    for name in df['piece']:
        color2 = (0,0,255)
        img = cv2.imread(join(piece_folder, name+".jpg"))
        x1 = df.loc[df['piece'] == name, 'X1'].iloc[0]
        y1 = df.loc[df['piece'] == name, 'Y1'].iloc[0] 
        x = int(x1)
        y = int(y1)
        xy = (x,y)
        cv2.circle(img,xy,2,color=color2,thickness=3)
        x1 = df.loc[df['piece'] == name, 'X2'].iloc[0]
        y1 = df.loc[df['piece'] == name, 'Y2'].iloc[0] 
        x = int(x1)
        y = int(y1)
        xy = (x,y)
        cv2.circle(img,xy,2,color=color2,thickness=3)
        x1 = df.loc[df['piece'] == name, 'X3'].iloc[0]
        y1 = df.loc[df['piece'] == name, 'Y3'].iloc[0] 
        x = int(x1)
        y = int(y1)
        xy = (x,y)
        cv2.circle(img,xy,2,color=color2,thickness=3)
        x1 = df.loc[df['piece'] == name, 'X4'].iloc[0]
        y1 = df.loc[df['piece'] == name, 'Y4'].iloc[0]  
        x = int(x1)
        y = int(y1)
        xy = (x,y)
        cv2.circle(img,xy,2,color=color2,thickness=3)
        cv2.imshow(name,img)
        
        cv2.destroyWindow(name)


def detect_side_images(df: pd.DataFrame,pieces_folder: str,sides_folder: str):
    try:

        for name in df['piece']:
            if(df.loc[df['piece'] == name, 'status'].iloc[0] == 'p'):
                img = cv2.imread(join(pieces_folder, name+".jpg"))
                out_dict = {}
                out_dict['name'] = name
                get_side_image(img,out_dict,df,name,sides_folder)
                df.loc[df['piece'] == name, 'status'] = 's'
                df.loc[df['piece'] == name, 'IO1'] = out_dict['in_out'][0]
                df.loc[df['piece'] == name, 'IO2'] = out_dict['in_out'][1]
                df.loc[df['piece'] == name, 'IO3'] = out_dict['in_out'][2]
                df.loc[df['piece'] == name, 'IO4'] = out_dict['in_out'][3]
                
        return df
    except Exception as e:
        out_dict['error'] = e

# function to detect side image for all pices bbased on the corners writen in df_pieces
def get_side_image(img,out_dict,df: pd.DataFrame,\
                   name: str,\
                    sides_folder: str):
    
    x1 = df.loc[df['piece'] == name, 'X1'].iloc[0]
    y1 = df.loc[df['piece'] == name, 'Y1'].iloc[0]
    x2 = df.loc[df['piece'] == name, 'X2'].iloc[0]
    y2 = df.loc[df['piece'] == name, 'Y2'].iloc[0]
    x3 = df.loc[df['piece'] == name, 'X3'].iloc[0]
    y3 = df.loc[df['piece'] == name, 'Y3'].iloc[0]
    x4 = df.loc[df['piece'] == name, 'X4'].iloc[0]
    y4 = df.loc[df['piece'] == name, 'Y4'].iloc[0]
    xy_array = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    # xy_array = out_dict['xy']
    # xy = order_corners(xy_array)
    xy = order_points_clockwise(xy_array)
    out_dict['xy'] = xy
    
    process_piece1(image=img,out_dict=out_dict,df_pieces=df)
    
    
    
    
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
    # df_sides = pd.read_csv('sides.csv')
    read_camera_images(page_number = 1, df=df_pieces)
    # df_pieces.to_csv('pieces.csv', index=False)
    # df_pieces = get_corners('pieces_threshold',df_pieces)
    
    # # show_image_with_corners('pieces_threshold',df_pieces)

    # df_pieces = detect_side_images(df_pieces,"pieces_threshold","sides")

    # df_pieces.to_csv('pieces.csv', index=False)
    # find_geometries()
    # transparent1()
    # df_pieces = get_corners('pieces_threshold',df_pieces)
    # df_pieces.to_csv('pieces.csv', index=False)
    find_geometries()
    
    
    # find_the_best_matchs()

if __name__ == "__main__":
    main()