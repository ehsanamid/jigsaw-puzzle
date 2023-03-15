from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import get_line_through_points, distance_point_line_squared, distance
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import math
# rotation agle for each side of horizontal and vertical piece
rotation_matrix = np.array([[0, 90, 180, -90], [180, -90, 0, 90]])

# rotate a list of point by a given angle and given pivot point and return \
# a list of rotated points
# def rotate_points(points, angle, pivot):
#     angle = np.radians(angle)
#     R = np.array([[np.cos(angle), -np.sin(angle)],
#                   [np.sin(angle), np.cos(angle)]])
#     # convert numpy array to list
#     points = np.array(np.dot(points - pivot, R) + pivot)
#     return points



def show_point(points):
    blank_image = np.zeros((600,600), np.uint8)
    for j in range(len(points)-1):
        pt1 = (points[j][0],points[j][1])
        pt2 = (points[j+1][0],points[j+1][1])
        cv2.line(blank_image, pt1, pt2, (255,255,255), 1)
        cv2.imshow("image",blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   


def rotate_points(points, angle, pivot):
    """Rotate a list of points by a given angle around a pivot point.

    Args:
        points (list of tuple): A list of points as (x, y) tuples.
        angle (float): The angle to rotate the points, in degrees.
        pivot (tuple): The pivot point as an (x, y) tuple.

    Returns:
        A list of rotated points as (x, y) tuples.
    """
    # Convert angle to radians
    radians = math.radians(angle)

    # Compute sin and cosine of angle
    cos = math.cos(radians)
    sin = math.sin(radians)

    # Translate pivot point to origin
    px, py = pivot
    translated = [(x - px, y - py) for x, y in points]

    # Rotate points around origin
    rotated = [[x * cos - y * sin, x * sin + y * cos] for x, y in translated]

    # Translate points back to original position
    result = [[int(round(x) + px), int(round(y) + py)] for x, y in rotated]


    return result
# return geometry of the points
def get_geometry(points):
    try:
        # get the center of the xy
        center = np.mean(points, axis=0)
        # get the distance between the center and the first point
        dist = distance(points[0], center)
        # get the angle between the center and the first point
        angle = np.arctan2(points[0][1] - center[1], points[0][0] - center[0])
        # convert the angle to degree
        angle = np.degrees(angle)
        # get the maximum x value
        max_x = max(points, key=lambda x: x[0])[0]
        critical_points = []
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
                    dist = distance(points[idx[j]], points[idx[j+1]])
                    
                    if dist < 2:
                        segment_len += 1
                    else:
                        blocks.append([start, segment_len])
                        start = idx[j+1]
                        segment_len = 0
                blocks.append([start, segment_len])        
            if (len(blocks) ==2):
                critical_points.append(blocks)
            
                
        # get points with maximum y and the index of the point
        max_y = max(points, key=lambda x: x[1])
        max_y_idx = points.index(max_y)
        count_max_y = points.count(max_y)
        


        # return the geometry
        return (center, dist, angle,max_y[0],count_max_y)
    except Exception as e:
        print(str(e))

def side_to_image(out_dict: dict, idx: int,points: list, filename: str):
    try:
        # show_point(points)
        oriatation = out_dict['in_out'][0]
        points = rotate_points(points, \
                               rotation_matrix[oriatation, idx], \
                                out_dict['xy'][idx])
        
        # blank_image = np.zeros((max_size(0),max_size(1),3), np.uint8)
        # minimum value of x and y
        minx = min(points, key=lambda x: x[0])[0]
        miny = min(points, key=lambda x: x[1])[1]
        # maximum value of x and y
        maxx = max(points, key=lambda x: x[0])[0]
        maxy = max(points, key=lambda x: x[1])[1]
        marg = 0
        # size of the image
        sizex, sizey = ((maxx - minx + marg*2), (maxy - miny+marg*2))
        blank_image = np.zeros((sizey, sizex, 3), np.uint8)
        # shift all points to (minx,miny) and add margin
        points = [[points[i][0] - minx + marg,points[i][1] - miny + marg] \
                  for i in range(len(points))]
        # draw the contour
        for i in range(len(points) -1 ):
            # index1 = i 
            # index2 = (i+1) 
            # pt1 = [points[index1][0] - minx + marg, points[index1][1] - miny + marg]
            # pt2 = [points[index2][0] - minx + marg, points[index2][1] - miny + marg]
            pt1 = points[i]
            pt2 = points[i+1]
            cv2.line(blank_image, pt1, pt2, (255, 255, 255), 1)
        
        
            
        # add "in" or "out" to the file name based on orientation
        if out_dict['in_out'][idx] == 0:
            filename = filename + "_in"
        else:
            filename = filename + "_out"
        cv2.imwrite(join('sides', filename +".jpg"),blank_image)
        # cv2.imshow("image",blank_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except Exception as e:
        print(str(e))





def shape_classification(out_dict,img,edges):
    
    try:
        threshhold = 30

        # get the center of the xy
        center = np.mean(out_dict['xy'],axis=0)


        lines = []  
        lines.append(get_line_through_points(out_dict['xy'][0],out_dict['xy'][1]))
        lines.append(get_line_through_points(out_dict['xy'][1],out_dict['xy'][2]))
        lines.append(get_line_through_points(out_dict['xy'][2],out_dict['xy'][3]))
        lines.append(get_line_through_points(out_dict['xy'][3],out_dict['xy'][0]))


        # class_image = np.zeros(img.shape, dtype='uint8')
        c_points = []

        for _edge in edges:
            d = [distance_point_line_squared(line, _edge) for line in lines]
            if np.min(d) < threshhold:
                ind = np.argmin(d)  
                c_points.append([_edge[0],_edge[1],ind,-1])
            else:
                c_points.append([_edge[0],_edge[1],-1,-1])
        
        list_length = len(c_points)
        # check any point is classified as -1
        while any([x[2] == -1 for x in c_points]):
            for i in range(list_length):
                if c_points[i][2] == -1:
                    ind1 = (i + list_length - 1) % list_length
                    ind2 = (i + 1) % list_length
                    if c_points[ind1][2] != -1:
                        c_points[i][3] = c_points[ind1][2]
                    elif c_points[ind2][2] != -1:
                        c_points[i][3] = c_points[ind2][2]
                else:
                    c_points[i][3] = c_points[i][2]
            for i in range(list_length):
                c_points[i][2] = c_points[i][3]
                c_points[i][3] = -1
        blank_image = []
        blank_image.append(np.zeros(img.shape, np.uint8))
        blank_image.append(np.zeros(img.shape, np.uint8))
        blank_image.append(np.zeros(img.shape, np.uint8))
        blank_image.append(np.zeros(img.shape, np.uint8))
        filename = out_dict['name'] 

        for classified_point in c_points:
            classified_point[3] = -1
        
        
        list_length = len(c_points)
        for j in range(list_length):
            if(( j == 0) and (c_points[j][2] != c_points[j+1][2]) or\
                ( j == list_length-1) and (c_points[j][2] != c_points[j-1][2])):
                c_points[j][3]= 0
            else:
                if ((c_points[j][2] != c_points[j-1][2]) and\
                    (c_points[j][2] != c_points[j+1][2])):
                    c_points[j][3]= 0
            
        # remove all rows that [3] is zero
        c_points = [x for x in c_points if x[3] != 0]    


        for i in range(4):
            while True:
                blocks = []
                # list_of_blocks = [] 
                list_length = len(c_points)
                for j in range(list_length):
                    if c_points[j][2] == i:
                        if( j == 0):
                            start_index = j
                        else:
                            if c_points[j][2] != c_points[j-1][2]:
                                start_index = j
                        if( j == list_length-1):
                            end_index = j
                            blocks.append([start_index,end_index])
                        else:
                            if c_points[j][2] != c_points[j+1][2]:
                                end_index = j
                                blocks.append([start_index,end_index])
                                
                
                # list_of_blocks.append(blocks)
                _point = []
                if(len(blocks)>1):
                    block1 = blocks[0]
                    block2 = blocks[1]
                    start_ind1 = block1[0]
                    end_ind1 = block1[1]
                    start_ind2 = block2[0]
                    end_ind2 = block2[1]
                    start_point1 = (c_points[start_ind1][0],c_points[start_ind1][1])
                    start_point2 = (c_points[start_ind2][0],c_points[start_ind2][1])
                    end_point1 = (c_points[end_ind1][0],c_points[end_ind1][1])
                    end_point2 = (c_points[end_ind2][0],c_points[end_ind2][1])

                    d1 = distance(start_point1,end_point2)
                    d2 = distance(start_point2,end_point1)
                    
                    if(d1<d2):
                        _point = c_points[block2[0]:block2[1]+1]
                        _point += c_points[block1[0]:block1[1]+1]
                        _point += c_points[0:block1[0]]
                        _point += c_points[block1[1]+1:block2[0]]
                        if(block2[1]+1 < list_length):
                            _point += c_points[block2[1]+1:list_length]
                    else:
                        _point = c_points[block1[0]:block1[1]+1]
                        _point += c_points[block2[0]:block2[1]+1]
                        _point += c_points[0:block1[0]]
                        _point += c_points[block1[1]+1:block2[0]]
                        if(block2[1]+1 < list_length):
                            _point += c_points[block2[1]+1:list_length]
                    c_points = _point
                else:
                    # _point = classified_points[blocks[0][0]:blocks[0][1]+1]
                    break

        four_sides_points = []

        four_sides_points.append([[x[0],x[1]] for x in c_points if x[2] == 0])
        four_sides_points.append([[x[0],x[1]] for x in c_points if x[2] == 1])
        four_sides_points.append([[x[0],x[1]] for x in c_points if x[2] == 2])
        four_sides_points.append([[x[0],x[1]] for x in c_points if x[2] == 3])
        in_out = []
        for i,points in enumerate(four_sides_points):
            head_point = 0
            head_point_index = -1
            
            for j in range(len(four_sides_points[i])):
                # find the maximum distance between points in points[i] and lines[i]
                
                d = distance_point_line_squared(lines[i], points[j])
                if (d > head_point):
                    head_point = d
                    head_point_index = j
            corner1 = out_dict['xy'][i]
            corner2 = out_dict['xy'][(i+1)%4]    
            
            # pt1 is middle point between corner1 and corner2
            pt1 = [(corner1[0]+corner2[0])/2,(corner1[1]+corner2[1])/2]
            d1 = distance(pt1,center)
            d2 = distance(points[head_point_index],center)
            if(d1<d2):
                in_out.append(1)
            else:
                in_out.append(0)
                
        out_dict['in_out'] = in_out
        out_dict['geometry'] = []
        for i,points in enumerate(four_sides_points):
            side_to_image(out_dict,i,points, filename+"_"+str(i+1))

         
        
        # for i in range(4):
        #     cv2.imwrite(join('sides', filename+"_"+str(i+1)+".jpg"), blank_image[i])  

                
        # return class_image
    except Exception as e:
        print(str(e))
         

####################################################################################################################
    



# function to return a piece of contour between two points
def get_piece_contour(contour, p1, p2):
    # get the index of the points in the contour
    idx1 = np.where((contour == p1).all(axis=2))[0][0]
    idx2 = np.where((contour == p2).all(axis=2))[0][0]
    # get the piece of contour
    if idx1 < idx2:
        piece = contour[idx1:idx2]
    else:
        piece = np.concatenate((contour[idx1:], contour[:idx2]))
    return piece

# function to show a piece of contour between two points
def show_piece_contour(contour, p1, p2):
    piece = get_piece_contour(contour, p1, p2)
    plt.plot(piece[:, 0, 0], piece[:, 0, 1], 'r-')
    plt.show()

def show_contour_piece(contour, filename: str, count:int, start, end):
    """Shows a piece of contour between two points."""
    contour = np.asarray(contour)
    start_idx = np.argmin(np.sum((contour - start)**2, axis=1))
    end_idx = np.argmin(np.sum((contour - end)**2, axis=1))
    contour_size = len(contour)
    # if end_idx < start_idx:
    #     start_idx, end_idx = end_idx, start_idx
    # plt.plot(contour[start_idx:end_idx+1, 0], contour[start_idx:end_idx+1, 1])
    if end_idx < start_idx:
        plt.plot(contour[start_idx:contour_size, 0], contour[start_idx:contour_size, 1])
        plt.plot(contour[0:end_idx+1, 0], contour[0:end_idx+1, 1])

    else:
        plt.plot(contour[start_idx:end_idx+1, 0], contour[start_idx:end_idx+1, 1])
    
    

def find_nearest_point(contour, point):
    """Finds the nearest point in a contour to a given point."""
    contour = np.asarray(contour)
    dists = np.sqrt(np.sum((contour - point)**2, axis=1))
    nearest_idx = np.argmin(dists)
    nearest_point = contour[nearest_idx]
    return nearest_point     

def get_max_min_x_y(points,indices):
    # Calculate the minimum and maximum x and y values
    minx = 100000
    miny = 100000
    maxx = 0
    maxy = 0
    for i in indices:
        if points[i][0] < minx:
            minx = points[i][0]
        if points[i][0] > maxx:
            maxx = points[i][0]
        if points[i][1] < miny:
            miny = points[i][1]
        if points[i][1] > maxy:
            maxy = points[i][1]

    # maxx = maxx
    # maxy = maxy
    # minx = minx
    # miny = miny
    return minx, miny, maxx, maxy

def get_side_size(minx: int,miny: int,maxx: int,maxy: int,margin: int=10):
 
    if( minx > margin):
        minx = minx - margin
    if( miny > margin):
        miny = miny - margin
    return ((maxx - minx + margin), (maxy - miny+margin))

'''def read_indicses_direction(points, start, end):
    """Reads the indices of a contour in a given direction."""
    buffer_size=len(points)
    # Initialize the circular buffer with zeros
    buffer = np.zeros((buffer_size, 2))

    # Keep track of the index of the next point to insert into the buffer
    index = 0

    # Iterate through the points and insert them into the buffer
    for point in points:
        buffer[index] = point
        index = (index + 1) % buffer_size

    start_idx = np.argmin(np.sum((points - start)**2, axis=1))
    end_idx = np.argmin(np.sum((points - end)**2, axis=1))
    
    # Calculate the indices of the points between p1 and p2
    if(end_idx < start_idx):
        start_idx, end_idx = end_idx, start_idx

    indices1 = np.arange(start_idx, end_idx + 1)
    indices2 = np.concatenate((np.arange(end_idx, buffer_size), \
        np.arange(start_idx + 1)))

    minx1,miny1,maxx1,maxy1 = get_max_min_x_y(buffer,indices1)
    minx2,miny2,maxx2,maxy2 = get_max_min_x_y(buffer,indices2)

    sizex1, sizey1 = get_side_size(minx1,miny1,maxx1,maxy1) 
    sizex2, sizey2 = get_side_size(minx2,miny2,maxx2,maxy2)

    if(sizex1 * sizey1 > sizex2 * sizey2):
        return minx2, miny2, maxx2, maxy2,sizex2,sizey2,indices2,buffer
    else:
        return minx1, miny1, maxx1, maxy1,sizex1,sizey1,indices1,buffer
    

def circular_buffer(points, filename: str, count:int, start, end):

    minx, miny, maxx, maxy,sizex, sizey,indices,buffer = \
        read_indicses_direction(points, start, end)
    
    # Create a new image with a black background
    image = np.zeros((sizey, sizex, 3), dtype=np.uint8)

    # Draw lines between the selected points on the image
    for i in range(len(indices) - 1):
        pt1 = (buffer[indices[i]][0].astype(int) - minx, \
            buffer[indices[i]][1].astype(int) - miny)
        pt2 = (buffer[indices[i+1]][0].astype(int) - minx, \
            buffer[indices[i+1]][1].astype(int) - miny)
        
        cv2.line(image, pt1, pt2, (255, 255, 255), 1)

    # Save the image to disk
    cv2.imwrite(join('sides', filename+"_"+str(count)+".jpg"), image)
'''



def compute_similarity(list1, list2):
    """
    Computes the Dynamic Time Warping similarity between two lists of points.
    
    Args:
    list1 (list): The first list of points.
    list2 (list): The second list of points.
    
    Returns:
    float: The DTW similarity score between the two lists of points.
    """
    # Compute the pairwise distances between the points
    distance_matrix = [[euclidean(x, y) for y in list2] for x in list1]
    
    # Compute the DTW distance and path
    distance, path = fastdtw(distance_matrix, dist=euclidean)
    
    # Compute the DTW similarity score
    similarity_score = 1 / (1 + distance)
    
    return similarity_score


# def read_indicses_direction(points, start, end,direction:int):
#     try:
#         """Reads the indices of a contour in a given direction."""
        
#         buffer_size=len(points)

#         start_idx = np.argmin(np.sum((points - start)**2, axis=1))
#         end_idx = np.argmin(np.sum((points - end)**2, axis=1))
        
#         # Calculate the indices of the points between p1 and p2
#         if(end_idx < start_idx):
#             start_idx, end_idx = end_idx, start_idx

#         indices1 = np.arange(start_idx, end_idx + 1)
#         indices2 = np.concatenate((np.arange(end_idx, buffer_size), \
#           np.arange(start_idx + 1)))

#         minx1,miny1,maxx1,maxy1 = get_max_min_x_y(points,indices1)
#         minx2,miny2,maxx2,maxy2 = get_max_min_x_y(points,indices2)

#         sizex1, sizey1 = get_side_size(minx1,miny1,maxx1,maxy1) 
#         sizex2, sizey2 = get_side_size(minx2,miny2,maxx2,maxy2)

#         if(sizex1 * sizey1 > sizex2 * sizey2):
#             return minx2, miny2, maxx2, maxy2,sizex2,sizey2,indices2
#         else:
#             return minx1, miny1, maxx1, maxy1,sizex1,sizey1,indices1
#     except Exception as e:
#         print(str(e))

# def circular_buffer(points, filename: str, count:int, start, end,direction:int=0):
#     try:
#         minx, miny, maxx, maxy,sizex, sizey,indices = \
#           read_indicses_direction(points, start, end,direction)
        
#         # Create a new image with a black background
#         image = np.zeros((sizey, sizex, 3), dtype=np.uint8)

#         # Draw lines between the selected points on the image
#         for i in range(len(indices) - 1):
#             pt1 = (points[indices[i]][0] - minx, points[indices[i]][1] - miny)
#             pt2 = (points[indices[i+1]][0] - minx, points[indices[i+1]][1] - miny)
            
#             cv2.line(image, pt1, pt2, (255, 255, 255), 1)

#         # Save the image to disk
#         cv2.imwrite(join('sides', filename+"_"+str(count)+".jpg"), image)
#     except Exception as e:
#         print(str(e))

# function to get list of points and maximu size and save the pints in an image
def contour_to_image(out_dict,points, filename: str):
    try:
        # blank_image = np.zeros((max_size(0),max_size(1),3), np.uint8)
        # minimum value of x and y
        minx = min(points, key=lambda x: x[0])[0]
        miny = min(points, key=lambda x: x[1])[1]
        # maximum value of x and y
        maxx = max(points, key=lambda x: x[0])[0]
        maxy = max(points, key=lambda x: x[1])[1]
        margin = 50
        # size of the image
        sizex, sizey = ((maxx - minx + margin*2), (maxy - miny+margin*2))

        blank_image = np.zeros((sizey, sizex, 3), np.uint8)

         

        '''for pt1 in points: 
            pt = [pt1[0] - minx + marg, pt1[1] - miny + marg]
            cv2.circle(img=blank_image, center=(pt[0], pt[1]), \
                radius=0, color=(255, 255, 255), thickness=-1)'''

        for i in range(len(out_dict['xy'])):
            out_dict['xy'][i] = [out_dict['xy'][i][0] - minx + margin, out_dict['xy'][i][1] - miny + margin]
        
        # lines = []

        
        # lines.append(get_line_through_points(out_dict['xy'][0],out_dict['xy'][1]))
        # lines.append(get_line_through_points(out_dict['xy'][1],out_dict['xy'][2]))
        # lines.append(get_line_through_points(out_dict['xy'][2],out_dict['xy'][3]))
        # lines.append(get_line_through_points(out_dict['xy'][3],out_dict['xy'][0]))


        with open(filename+".csv", 'w') as file:
            for i in range(4):
                file.write(f"{out_dict['xy'][i][0]}, {out_dict['xy'][i][1]}\n")
            
            # for i in range(4):
            #     file.write(f"{lines[i][0]}, {lines[i][1]}, {lines[i][2]}\n")
            new_points = []
            # draw the contour
            for i in range(len(points) ):
                index1 = i % len(points)
                index2 = (i+1) % len(points)
                
                pt1 = [points[index1][0] - minx + margin, \
                       points[index1][1] - miny + margin]
                pt2 = [points[index2][0] - minx + margin, \
                       points[index2][1] - miny + margin]
                file.write(f"{pt1[0]}, {pt1[1]}\n")
                new_points.append(pt1)
                cv2.line(blank_image, pt1, pt2, (255, 255, 255), 1)

        ret, gray = cv2.threshold(blank_image, 128, 255, cv2.THRESH_BINARY) 

        # cv2.drawContours(blank_image, new_points, -1, (255,255,255), 1)
        cv2.imwrite(join('contours', filename +".jpg"),gray)
        return blank_image,new_points
    except Exception as e:
        print(str(e))




def process_piece1(image,out_dict,df_pieces):
    try:
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray1, 128, 255, cv2.THRESH_BINARY_INV) 

        # xy = out_dict['xy']
        edged = cv2.Canny(gray,30,200)
    

        # get the countours of the piece
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, \
                                               cv2.CHAIN_APPROX_TC89_KCOS)
        
        # find the closes point in the contour to a point
        contour = max(contours, key=cv2.contourArea)
        contour_list = []
        for c in contour:
            # val = c[0].tolist()
            # if val not in contour_list:
            contour_list.append(c[0].tolist())

        

        new_img,new_points = contour_to_image(out_dict,points=contour_list,\
                                               filename=out_dict['name'])

        shape_classification(out_dict,new_img,new_points)

       

        
        
        '''p1 = find_nearest_point(contour_list, (xy[0][0],xy[0][1]))
        p2 = find_nearest_point(contour_list, (xy[1][0],xy[1][1]))
        p3 = find_nearest_point(contour_list, (xy[2][0],xy[2][1]))
        p4 = find_nearest_point(contour_list, (xy[3][0],xy[3][1]))

        circular_buffer(points=contour_list,\
                    filename=out_dict['name'],\
                    count=1, start=p1, end=p2)
        circular_buffer(points=contour_list,\
                    filename=out_dict['name'],\
                    count=2, start=p2, end=p3)
        circular_buffer(points=contour_list,\
                    filename=out_dict['name'],\
                    count=3, start=p3, end=p4)
        circular_buffer(points=contour_list,\
                    filename=out_dict['name'],\
                    count=4, start=p4, end=p1)'''

        
        
        

        
    except Exception as e:
        out_dict['error'] = e
    
    # finally:
    #     return gray




def find_white_pixels(img_path):
    # Read the image
    img = cv2.imread(img_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to get white pixels
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Find white pixel coordinates
    white_pixels = np.where(thresh == 255)
    coords = [[x, y] for x, y in zip(white_pixels[1], white_pixels[0])] # reverse coordinates to match (x,y) format
# Sort coordinates based on x and y
    sorted_coords = sorted(coords, key=lambda c: (c[0], c[1]))

    return sorted_coords


def get_image_geometry(filename: str):
    points = find_white_pixels(img_path = filename)
    # return geometry of the points
    geometry = get_geometry(points)
    return geometry