
import os
from os.path import join
import numpy as np
import pandas as pd
import cv2
from piece import Piece, SideShape, ShapeStatus
import math
import utils




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

def get_geometry(points):
    try:
        geometry = {}

        # remove the first 5 points and the last 5 points
        points = points[5:-5]

        # distance between the first point and the last point
        width = utils.distance(points[0], points[-1])
        if(width == 0):
            geometry['Width'] = 1000
        else:
            geometry["Width"] = width
    

        #get the line between the first point and the last point
        line = utils.get_line_through_points(points[0], points[-1])

        # find the point that has the maximum distance from the line
        max_dist = 0
        max_point = []

        # put disatnce from line to each point in a list
        dist = [utils.distance_point_line_squared(a_b_c=line, x0_y0=point) for point in points] 

        # find the max distance and its index
        max_dist = max(dist)
        
        # return list of indexes of max_dist
        max_index = [i for i, j in enumerate(dist) if j == max_dist]

        geometry['head_flatness'] = len(max_index)

        # find the middle point of the max_dist points
        max_point = points[int(len(max_index)/2)]

        geometry["Height"] = max_dist
        # find the intercept point of ortagonal line from max_point and line
        x1,y1 = utils.find_intersection(max_point[0], max_point[1],\
            line[0], line[1], line[2])

        symetry = utils.distance(points[0], (x1,y1)) / width
        geometry['symetry'] = symetry

        inter = utils.find_lines_interpolate(points_list=points)
        geometry["m"] = inter[0]
        geometry["c"] = inter[1]
        
       
        
        """ critical_points = []
        for i in range(max_x):
            # return index of points that have x value equal to i
            idx = [j for j, x in enumerate(points) if x[0] == i]
            # if there is more than one point with x value equal to i
            blocks = []
            start = 0
            segment_len = 0
            if len(idx) > 1:
                start = idx[0]
                segment_len = 0
                for j in range(len(idx)-1):
                    # get the distance between the two points
                    dist = points[idx[j+1]][1] - points[idx[j]][1]
                    
                    if dist < 2:
                        segment_len += 1
                    else:
                        blocks.append([start, segment_len])
                        start = idx[j+1]
                        segment_len = 0
                blocks.append([start, segment_len])        
            if (len(blocks) ==2):
                critical_points.append(blocks)
            
        for blocks in critical_points:
            for block in blocks:
                start_idx = block[0]
                block_len = block[1]
                X1 = points[start_idx][0]
                Y1 = points[start_idx][1]
                Y2 = points[start_idx + block_len][1]
                geometry.append(X1)
                geometry.append(Y1)
                # geometry.append(Y2)
                # geometry.append(Y2 - Y1)        
 """
        # return the geometry
        return geometry
    except Exception as e:
        print(str(e))
        return None
         




def find_geometries(df: pd.DataFrame):
    try:
        for index, row in df.iterrows():
            sidename = row['Side']
            file_name = join('sides', sidename+".csv")
            f = open(file_name,"r")
            # read list of x and y from the file and put them in points_list
            points_list = []
            for line in f:
                x,y = line.split(",")
                points_list.append([int(x),int(y)])
            f.close()
            geometry = get_geometry(points_list)
            if(geometry is None):
                continue
            
            df.loc[index, 'Width'] = geometry['Width']
            df.loc[index, 'Height'] = geometry['Width']
            df.loc[index, 'symetry'] = geometry['symetry']
            df.loc[index, 'm'] = geometry['m']
            df.loc[index, 'c'] = geometry['c']
            df.to_csv("sides.csv", index=False)


    except Exception as e:
        print(str(e))




def euclidean_distance(vec1, vec2):
    """Calculates the Euclidean distance between two vectors."""
    squared_diff = [(x - y) ** 2 for x, y in zip(vec1, vec2)]
    return math.sqrt(sum(squared_diff))

def similarity_score(obj1, obj2):
    """Calculates the similarity score between two objects."""
    distance = euclidean_distance(obj1, obj2)
    similarity = 1 / (1 + distance)
    return similarity

# Example list of objects with numerical properties
objects = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Calculate similarity between all object pairs
num_objects = len(objects)
similarity_matrix = [[0] * num_objects for _ in range(num_objects)]

for i in range(num_objects):
    for j in range(i + 1, num_objects):
        obj1 = objects[i]
        obj2 = objects[j]
        similarity = similarity_score(obj1, obj2)
        similarity_matrix[i][j] = similarity
        similarity_matrix[j][i] = similarity

# Print the similarity matrix
for row in similarity_matrix:
    print(row)

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
def extract(page_number: int, folder_name: str, df: pd.DataFrame):
    filenames = os.listdir(folder_name)
    filenames.sort()
    i = 1
    j = 1
    new_record_added = False
    for filename in filenames:
        piece_file_name = f"Page_{page_number:04d}_{i}_{j}"
        j = j + 1
        if(j > 7):
            j =1
            i = i+1
        if (df['piece'].eq(piece_file_name)).any():
            continue
        # status of a piece
        # status = df.loc[df['piece'] == piece_file_name, 'status'].iloc[0]
        status = ShapeStatus.Piece
        piece = Piece(piece_file_name)
        if(piece.camera_image_to_piece(filename,folder_name,status)):  
            print(f"{piece_file_name} added\n")
            new_row = pd.DataFrame({'piece':piece_file_name, \
            'status':str(status.value), \
            'X1':0, \
            'Y1':0, \
            'IO1':str(SideShape.UNDEFINED.value), \
            'X2':0, \
            'Y2':0, \
            'IO2':str(SideShape.UNDEFINED.value), \
            'X3':0, \
            'Y3':0, \
            'IO3':str(SideShape.UNDEFINED.value), \
            'X4':0, \
            'Y4':0, \
            'IO4':str(SideShape.UNDEFINED.value), \
                },index=[0])
            df = pd.concat([new_row,df.loc[:]]).reset_index(drop=True)
            new_record_added = True
    if(new_record_added):
        df.to_csv("pieces.csv", index=False)

    return df
   
def get_corners_from_pointlist(df: pd.DataFrame):
    try:
        # loop through all records in df dataframe
        for index, row in df.iterrows():
            piecename = row['piece']
            piece = Piece(piecename)
            piece.get_corners_from_pointlist(420, 420)
                    
    except Exception as e:
        print(str(e))


def find_corners(df: pd.DataFrame,width: int, height: int):
    try:
        # loop through all records in df dataframe
        for index, row in df.iterrows():
            piecename = row['piece']
            status = ShapeStatus(row['status'])
            if(status == ShapeStatus.Piece):
                piece = Piece(piecename)
                if(piece.find_contour(width=width, height=height)):
                    # update datafare row
                    df.loc[index, 'status'] = ShapeStatus.Side.value
                    df.loc[index, 'X1'] = piece.corners[0][0]
                    df.loc[index, 'Y1'] = piece.corners[0][1]
                    # df.loc[index, 'IO1'] = piece.in_out[0].value
                    df.loc[index, 'X2'] = piece.corners[1][0]
                    df.loc[index, 'Y2'] = piece.corners[1][1]
                    # df.loc[index, 'IO2'] = piece.in_out[1].value
                    df.loc[index, 'X3'] = piece.corners[2][0]
                    df.loc[index, 'Y3'] = piece.corners[2][1]
                    # df.loc[index, 'IO3'] = piece.in_out[2].value
                    df.loc[index, 'X4'] = piece.corners[3][0]
                    df.loc[index, 'Y4'] = piece.corners[3][1]
                    # df.loc[index, 'IO4'] = piece.in_out[3].value
                    df.to_csv("pieces.csv", index=False)
                    print(f"{piecename} corners found\n")
            
        
        
        # return df_pieces
    except Exception as e:
        print(str(e))

def show_corners(df: pd.DataFrame):
    try:
        # loop through all records in df dataframe
        for index, row in df.iterrows():
            piecename = row['piece']
            status = ShapeStatus(row['status'])
            X1 = df.loc[index, 'X1']
            Y1 = df.loc[index, 'Y1']
            X2 = df.loc[index, 'X2']
            Y2 = df.loc[index, 'Y2']
            X3 = df.loc[index, 'X3']
            Y3 = df.loc[index, 'Y3']
            X4 = df.loc[index, 'X4']
            Y4 = df.loc[index, 'Y4']
            if(status == ShapeStatus.Edge):
                piece = Piece(piecename)
                piece.show_corners(X1,Y1,X2,Y2,X3,Y3,X4,Y4)
                    
        
        # return df_pieces
    except Exception as e:
        print(str(e))

def find_shape_in_out(df: pd.DataFrame):
    try:
        # loop through all records in df dataframe
        for index, row in df.iterrows():
            piecename = row['piece']
            status = ShapeStatus(row['status'])
            if(status == ShapeStatus.Side):
                X1 = df.loc[index, 'X1']
                Y1 = df.loc[index, 'Y1']
                X2 = df.loc[index, 'X2']
                Y2 = df.loc[index, 'Y2']
                X3 = df.loc[index, 'X3']
                Y3 = df.loc[index, 'Y3']
                X4 = df.loc[index, 'X4']
                Y4 = df.loc[index, 'Y4']
                piece = Piece(piecename)
                if(piece.clasification(X1,Y1,X2,Y2,X3,Y3,X4,Y4)):
                    # update datafare row
                    df.loc[index, 'status'] = ShapeStatus.Edge.value
                    df.loc[index, 'X1'] = piece.corners[0][0]
                    df.loc[index, 'Y1'] = piece.corners[0][1]
                    df.loc[index, 'IO1'] = piece.in_out[0].value
                    df.loc[index, 'X2'] = piece.corners[1][0]
                    df.loc[index, 'Y2'] = piece.corners[1][1]
                    df.loc[index, 'IO2'] = piece.in_out[1].value
                    df.loc[index, 'X3'] = piece.corners[2][0]
                    df.loc[index, 'Y3'] = piece.corners[2][1]
                    df.loc[index, 'IO3'] = piece.in_out[2].value
                    df.loc[index, 'X4'] = piece.corners[3][0]
                    df.loc[index, 'Y4'] = piece.corners[3][1]
                    df.loc[index, 'IO4'] = piece.in_out[3].value
                    df.to_csv("pieces.csv", index=False)
                    print(f"{piecename} corners found\n")
            
        
        
        # return df_pieces
    except Exception as e:
        print(str(e))



def find_sides(df: pd.DataFrame):
    try:
        # loop through all records in df dataframe
        for index, row in df.iterrows():
            piecename = row['piece']
            status = ShapeStatus(row['status'])
            if(status == ShapeStatus.Side):
                piece = Piece(piecename)
                if(piece.find_corners()):
                    # update datafare row
                    df.loc[index, 'status'] = ShapeStatus.Edge.value
                    df.loc[index, 'X1'] = piece.corners[0][0]
                    df.loc[index, 'Y1'] = piece.corners[0][1]
                    df.loc[index, 'IO1'] = piece.in_out[0].value
                    df.loc[index, 'X2'] = piece.corners[1][0]
                    df.loc[index, 'Y2'] = piece.corners[1][1]
                    df.loc[index, 'IO2'] = piece.in_out[1].value
                    df.loc[index, 'X3'] = piece.corners[2][0]
                    df.loc[index, 'Y3'] = piece.corners[2][1]
                    df.loc[index, 'IO3'] = piece.in_out[2].value
                    df.loc[index, 'X4'] = piece.corners[3][0]
                    df.loc[index, 'Y4'] = piece.corners[3][1]
                    df.loc[index, 'IO4'] = piece.in_out[3].value
                    df.to_csv("pieces.csv", index=False)
                    print(f"{piecename} corners found\n")
            
        
        
        # return df_pieces
    except Exception as e:
        print(str(e))




# read all images in a folder and get the maximum size of the image
def get_max_size(folder_name):
    max_width = 0
    max_height = 0
    for filename in os.listdir(folder_name):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder_name, filename))
            height, width, channels = img.shape
            if(width > max_width):
                max_width = width
            if(height > max_height):
                max_height = height
    return max_width, max_height











def main():
    
    df_pieces = pd.read_csv('pieces.csv')
    df_sides = pd.read_csv('sides.csv')

    df_pieces = extract(page_number = 1,folder_name = "cam01", df=df_pieces)
    df_pieces = extract(page_number = 2,folder_name = "cam02", df=df_pieces)
    df_pieces = extract(page_number = 3,folder_name = "cam03", df=df_pieces)
    df_pieces = extract(page_number = 4,folder_name = "cam04", df=df_pieces)
    df_pieces = extract(page_number = 5,folder_name = "cam05", df=df_pieces)
    df_pieces = extract(page_number = 6,folder_name = "cam06", df=df_pieces)
    df_pieces = extract(page_number = 7,folder_name = "cam07", df=df_pieces)
    df_pieces = extract(page_number = 8,folder_name = "cam08", df=df_pieces)
    df_pieces = extract(page_number = 9,folder_name = "cam09", df=df_pieces)
    df_pieces = extract(page_number = 10,folder_name = "cam10", df=df_pieces)
    df_pieces = extract(page_number = 11,folder_name = "cam11", df=df_pieces)
    df_pieces = extract(page_number = 12,folder_name = "cam12", df=df_pieces)
    df_pieces = extract(page_number = 13,folder_name = "cam13", df=df_pieces)
    df_pieces = extract(page_number = 14,folder_name = "cam14", df=df_pieces)
    df_pieces = extract(page_number = 15,folder_name = "cam15", df=df_pieces)

    # max_width, max_height = get_max_size(folder_name="contours")

    get_corners_from_pointlist(df=df_pieces)
    find_corners(df=df_pieces,width=420, height=420)

    # find_shape_in_out(df=df_pieces)

    
    
    # df_pieces.to_csv('pieces.csv', index=False)
    # df_pieces = get_corners('pieces_threshold',df_pieces)
    
    # # show_image_with_corners('pieces_threshold',df_pieces)

    # df_pieces = detect_side_images(df_pieces,"pieces_threshold","sides")

    # df_pieces.to_csv('pieces.csv', index=False)
    # find_geometries()
    # transparent1()
    # df_pieces = get_corners('pieces_threshold',df_pieces)
    # df_pieces.to_csv('pieces.csv', index=False)
    #show_corners(df_pieces)
    find_geometries(df_sides)
    
    
    # find_the_best_matchs()

if __name__ == "__main__":
    main()