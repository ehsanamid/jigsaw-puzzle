import os
from os.path import join
import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial
import cv2
import skimage
from sklearn.cluster import KMeans
from utils import get_line_through_points, distance_point_line_squared, distance_point_line_signed, rotate,distance,intersection_point

_corner_indexes = [(0, 1), (1, 3), (3, 2), (0, 2)]


def compute_barycentre(thresh, value=0):
    """
    Given the segmented puzzle piece, compute its barycentre.
    """
    idx_shape = np.where(thresh == value)
    return [int(np.round(coords.mean())) for coords in idx_shape]

def compute_minmax_xy(thresh):
    """
    Given the thresholded image, compute the minimum and maximum x and y 
    coordinates of the segmented puzzle piece.
    """
    idx_shape = np.where(thresh == 0)
    return [np.array([coords.min(), coords.max()]) for coords in idx_shape]


def segment_piece(image, bin_threshold=128):
    """
    Apply segmentation of the image by simple binarization
    """
    return cv2.threshold(image, bin_threshold, 255, cv2.THRESH_BINARY)[1]

    
def extract_piece(thresh):

    # Here we build a square image centered on the blob (piece of the puzzle).
    # The image is constructed large enough to allow for piece rotations. 
    
    minmax_y, minmax_x = compute_minmax_xy(thresh)

    ly, lx = minmax_y[1] - minmax_y[0], minmax_x[1] - minmax_x[0]
    size = int(max(ly, lx) * np.sqrt(2))

    x_extract = thresh[minmax_y[0]:minmax_y[1] + 1, minmax_x[0]:minmax_x[1] + 1]
    ly, lx = x_extract.shape

    xeh, xew = x_extract.shape
    x_copy = np.full((size, size), 255, dtype='uint8')
    sy, sx = size // 2 - ly // 2, size // 2 - lx // 2

    x_copy[sy: sy + ly, sx: sx + lx] = x_extract
    thresh = x_copy
    thresh = 255 - thresh
    return thresh


def prune_lines_by_voting(lines, angle_threshold=5):
    
    accumulator = np.zeros(45)
    angles = lines[:, 1] * 180 / np.pi
    angles[angles >= 135] = 180 - angles[angles >= 135]
    angles[angles >= 90] -= 90
    angles[angles >= 45] = 90 - angles[angles >= 45]

    for angle, weight in zip(angles, np.linspace(1, 0.5, len(lines))):

        angle = int(np.round(angle))

        def add(a, w):
            if a >= 0 and a < len(accumulator):
                accumulator[a] += w

        add(angle - 3, weight * 0.1)
        add(angle - 2, weight * 0.5)
        add(angle - 1, weight * 0.8)
        add(angle, weight)
        add(angle + 1, weight * 0.8)
        add(angle + 2, weight * 0.5)
        add(angle + 3, weight * 0.1)

    # print accumulator
    best_angle = np.argmax(accumulator)
    print( 'best angle', best_angle)
    return lines[np.abs(angles - best_angle) <= angle_threshold]


def compute_mean_line(lines, debug=False):
    
    if len(lines) == 1:
        return lines[0]
    
    neg_idx = np.where(lines[:, 0] < 0)
    lines = lines.copy()
    lines[neg_idx, 0] = np.abs(lines[neg_idx, 0])
    lines[neg_idx, 1] = lines[neg_idx, 1] - np.pi
    
    weights = np.linspace(1.0, 0.5, len(lines))
    
    rhos = np.abs(lines[:, 0])
    mean_rho, std_rho = np.mean(rhos), np.std(rhos)
    
    gaussian_weigthts = np.array([scipy.stats.norm(mean_rho, std_rho).pdf(r) for r in rhos])    
    weights *= gaussian_weigthts
    
    sum_weights = np.sum(weights)
   
    # Compute weighted sum
    m_rho = np.sum(rhos * weights) / sum_weights
    
    sines, cosines = np.sin(lines[:, 1]), np.cos(lines[:, 1])

    m_sine = np.sum(sines * weights) / sum_weights
    m_cosine = np.sum(cosines * weights) / sum_weights
    m_theta = np.arctan2(m_sine, m_cosine)
    
    
    if debug:
        #print(correction_factor)
        print(weights)
        print()
    
    return np.array([m_rho, m_theta])



def line_intersection(line1, line2):
    
    # Solve the linear system that computes the intersection between
    # two lines, each one defined as a tuple (rho, theta) (the result comes from Hough lines)
    # If the lines have the same theta (parallel lines), a None result is returned
    
    rho1, theta1 = line1
    rho2, theta2 = line2

    if theta1 == theta2:
        return None, None

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    
    return x0, y0

#def compute_intersections(mean_lines, (h, w)):
def compute_intersections(mean_lines, h_w :tuple):    
    intersections = []
    h = h_w[0]
    w = h_w[1]
    for i, line_i in enumerate(mean_lines):
        for j, line_j in enumerate(mean_lines[i+1:], start=i+1):

            x0, y0 = line_intersection(line_i, line_j)

            if x0 >= 0 and y0 >= 0 and x0 < w and y0 < h:
                intersections.append([x0, y0])

    return np.array(intersections)


#def corner_detection(edges, intersections, (xb, yb), rect_size=50, show=False):
def corner_detection(edges, intersections, xb_yb : tuple, rect_size=50, show=False):

    # Find corners by taking the highest distant point from a 45 degrees inclined line
    # inside a squared ROI centerd on the previously found intersection point.
    # Inclination of the line depends on which corner we are looking for, and is
    # computed based on the position of the barycenter of the piece.

    corners = []
    xb = xb_yb[0]
    yb = xb_yb[1]
    for idx, intersection in enumerate(intersections):
            
        xi, yi = intersection

        m = -1 if (yb - yi)*(xb - xi) > 0 else 1
        y0 = 0 if yb < yi else 2*rect_size
        x0 = 0 if xb < xi else 2*rect_size

        a, b, c = m, -1, -m*x0 + y0

        rect = edges[yi - rect_size: yi + rect_size, xi - rect_size: xi + rect_size].copy()

        edge_idx = np.nonzero(rect)
        if len(edge_idx[0]) > 0:
            distances = [(a*edge_x + b*edge_y + c)**2 for edge_y, edge_x in zip(*edge_idx)]
            corner_idx = np.argmax(distances)

            rect_corner = np.array((edge_idx[1][corner_idx], edge_idx[0][corner_idx]))
            offset_corner = rect_corner - rect_size
            real_corner = intersection + offset_corner

            corners.append(real_corner)
        else:
            # If the window is completely black I can make no assumption: I keep the same corner
            corners.append(intersection)

        if show:
            plt.subplot(220 + idx + 1)
            cv2.circle(rect, tuple(rect_corner), 5, 128)
            
            plt.title("{0} | {1}".format(intersection, (x0, y0)))
            plt.imshow(rect)
    
    if show:
        plt.show()
        
    return corners


def order_corners1(corners):
    # Sort corners in increasing order of x-coordinate
    corners = sorted(corners,key=lambda corner: corner[0])
    
    # Split sorted corners into top and bottom halves
    top_corners, bottom_corners = corners[:2], corners[2:]
    
    # Sort top and bottom halves separately in increasing order of y-coordinate
    top_corners.sort(key=lambda corner: corner[1])
    bottom_corners.sort(key=lambda corner: corner[1])
    
    # Combine sorted top and bottom halves into final sorted list of corners
    sorted_corners = top_corners + bottom_corners
    
    return sorted_corners


def order_corners(corners):
    try:
        corners = sorted(corners,key=lambda k: k[0] + k[1])
        antidiag_corners = sorted(corners[2:4] , reverse= True  , key=lambda k: k[1])
        corners[2:4] = antidiag_corners
        return corners
    except ValueError:
        print(ValueError)
        return None

def compute_line_params(corners):
    return [get_line_through_points(corners[i1], corners[i2]) for i1, i2 in _corner_indexes]


def side_to_image(points, filename: str):
    try:
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
        # draw the contour
        for i in range(len(points) -1 ):
            index1 = i 
            index2 = (i+1) 
            pt1 = [points[index1][0] - minx + marg, points[index1][1] - miny + marg]
            pt2 = [points[index2][0] - minx + marg, points[index2][1] - miny + marg]
            cv2.line(blank_image, pt1, pt2, (255, 255, 255), 1)

        cv2.imwrite(join('sides', filename +".jpg"),blank_image)

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


        class_image = np.zeros(img.shape, dtype='uint8')
        classified_points = []

        for _edge in edges:
            d = [distance_point_line_squared(line, _edge) for line in lines]
            if np.min(d) < threshhold:
                ind = np.argmin(d)  
                classified_points.append([_edge[0],_edge[1],ind,-1])
            else:
                classified_points.append([_edge[0],_edge[1],-1,-1])
        
        list_length = len(classified_points)
        # check any point is classified as -1
        while any([x[2] == -1 for x in classified_points]):
            for i in range(list_length):
                if classified_points[i][2] == -1:
                    ind1 = (i + list_length - 1) % list_length
                    ind2 = (i + 1) % list_length
                    if classified_points[ind1][2] != -1:
                        classified_points[i][3] = classified_points[ind1][2]
                    elif classified_points[ind2][2] != -1:
                        classified_points[i][3] = classified_points[ind2][2]
                else:
                    classified_points[i][3] = classified_points[i][2]
            for i in range(list_length):
                classified_points[i][2] = classified_points[i][3]
                classified_points[i][3] = -1
        blank_image = []
        blank_image.append(np.zeros(img.shape, np.uint8))
        blank_image.append(np.zeros(img.shape, np.uint8))
        blank_image.append(np.zeros(img.shape, np.uint8))
        blank_image.append(np.zeros(img.shape, np.uint8))
        filename = out_dict['name'] 

        for classified_point in classified_points:
            classified_point[3] = -1
        
        
        list_length = len(classified_points)
        for j in range(list_length):
            if(( j == 0) and (classified_points[j][2] != classified_points[j+1][2]) or\
                ( j == list_length-1) and (classified_points[j][2] != classified_points[j-1][2])):
                classified_points[j][3]= 0
            else:
                if ((classified_points[j][2] != classified_points[j-1][2]) and\
                    (classified_points[j][2] != classified_points[j+1][2])):
                    classified_points[j][3]= 0
            
        # remove all rows that [3] is zero
        classified_points = [x for x in classified_points if x[3] != 0]    


        for i in range(4):
            while True:
                blocks = []
                list_of_blocks = [] 
                list_length = len(classified_points)
                for j in range(list_length):
                    if classified_points[j][2] == i:
                        if( j == 0):
                            start_index = j
                        else:
                            if classified_points[j][2] != classified_points[j-1][2]:
                                start_index = j
                        if( j == list_length-1):
                            end_index = j
                            blocks.append([start_index,end_index])
                        else:
                            if classified_points[j][2] != classified_points[j+1][2]:
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
                    start_point1 = (classified_points[start_ind1][0],classified_points[start_ind1][1])
                    start_point2 = (classified_points[start_ind2][0],classified_points[start_ind2][1])
                    end_point1 = (classified_points[end_ind1][0],classified_points[end_ind1][1])
                    end_point2 = (classified_points[end_ind2][0],classified_points[end_ind2][1])

                    d1 = distance(start_point1,end_point2)
                    d2 = distance(start_point2,end_point1)
                    
                    if(d1<d2):
                        _point = classified_points[block2[0]:block2[1]+1]
                        _point += classified_points[block1[0]:block1[1]+1]
                        _point += classified_points[0:block1[0]]
                        _point += classified_points[block1[1]+1:block2[0]]
                        if(block2[1]+1 < list_length):
                            _point += classified_points[block2[1]+1:list_length]
                    else:
                        _point = classified_points[block1[0]:block1[1]+1]
                        _point += classified_points[block2[0]:block2[1]+1]
                        _point += classified_points[0:block1[0]]
                        _point += classified_points[block1[1]+1:block2[0]]
                        if(block2[1]+1 < list_length):
                            _point += classified_points[block2[1]+1:list_length]
                    classified_points = _point
                else:
                    # _point = classified_points[blocks[0][0]:blocks[0][1]+1]
                    break

        points = []

        points.append([[x[0],x[1]] for x in classified_points if x[2] == 0])
        points.append([[x[0],x[1]] for x in classified_points if x[2] == 1])
        points.append([[x[0],x[1]] for x in classified_points if x[2] == 2])
        points.append([[x[0],x[1]] for x in classified_points if x[2] == 3])
        in_out = []
        for i in range(4):
            head_point = 0
            head_point_index = -1
            
            for j in range(len(points[i])):
                # find the maximum distance between points in points[i] and lines[i]
                
                d = distance_point_line_squared(lines[i], points[i][j])
                if (d > head_point):
                    head_point = d
                    head_point_index = j
            corner1 = out_dict['xy'][i]
            corner2 = out_dict['xy'][(i+1)%4]    
            pt1 = intersection_point(line1_point1=corner1,\
                                    line1_point2=corner2,\
                                    line2_point1=points[i][head_point_index],\
                                    line2_point2=center)
            d1 = distance(pt1,center)
            d2 = distance(points[i][head_point_index],center)
            if(d1<d2):
                in_out.append(1)
            else:
                in_out.append(0)
                
        out_dict['in_out'] = in_out
        for i in range(4):
            side_to_image(points[i], filename+"_"+str(i+1))

        # for i in range(4):
        #     for j in range(len(points[i])-1):
        #         pt1 = (points[i][j][0],points[i][j][1])
        #         pt2 = (points[i][j+1][0],points[i][j+1][1])
        #         cv2.line(blank_image[i], pt1, pt2, (255,255,255), 1)
        #         cv2.imshow(filename+"_"+str(i+1),blank_image[i])
            
        
            
                
            
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()    
        
        # for i in range(4):
        #     cv2.imwrite(join('sides', filename+"_"+str(i+1)+".jpg"), blank_image[i])  

                
        # return class_image
    except Exception as e:
        print(str(e))






        
    

  

####################################################################################################################
    

def get_default_params():
    
    
    side_extractor_default_values = {
        'before_segmentation_func': partial(cv2.medianBlur, ksize=5),
        'bin_threshold': 130,
        'after_segmentation_func': None,
        'scale_factor': 0.5,
        'harris_blocksize': 5,
        'harris_ksize': 5,
        'corner_nsize': 5,
        'corner_score_threshold': 0.2,
        'corner_minmax_threshold': 100,
        'corner_refine_rect_size': 5,
        'edge_erode_size': 3,
        'shape_classification_distance_threshold': 100,
        'shape_classification_nhs': 5,
        'inout_distance_threshold': 5
    }
    
    return side_extractor_default_values.copy()

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
    indices2 = np.concatenate((np.arange(end_idx, buffer_size), np.arange(start_idx + 1)))

    minx1,miny1,maxx1,maxy1 = get_max_min_x_y(buffer,indices1)
    minx2,miny2,maxx2,maxy2 = get_max_min_x_y(buffer,indices2)

    sizex1, sizey1 = get_side_size(minx1,miny1,maxx1,maxy1) 
    sizex2, sizey2 = get_side_size(minx2,miny2,maxx2,maxy2)

    if(sizex1 * sizey1 > sizex2 * sizey2):
        return minx2, miny2, maxx2, maxy2,sizex2,sizey2,indices2,buffer
    else:
        return minx1, miny1, maxx1, maxy1,sizex1,sizey1,indices1,buffer
    

def circular_buffer(points, filename: str, count:int, start, end):

    minx, miny, maxx, maxy,sizex, sizey,indices,buffer = read_indicses_direction(points, start, end)
    
    # Create a new image with a black background
    image = np.zeros((sizey, sizex, 3), dtype=np.uint8)

    # Draw lines between the selected points on the image
    for i in range(len(indices) - 1):
        pt1 = (buffer[indices[i]][0].astype(int) - minx, buffer[indices[i]][1].astype(int) - miny)
        pt2 = (buffer[indices[i+1]][0].astype(int) - minx, buffer[indices[i+1]][1].astype(int) - miny)
        
        cv2.line(image, pt1, pt2, (255, 255, 255), 1)

    # Save the image to disk
    cv2.imwrite(join('sides', filename+"_"+str(count)+".jpg"), image)
'''

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
#         indices2 = np.concatenate((np.arange(end_idx, buffer_size), np.arange(start_idx + 1)))

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
#         minx, miny, maxx, maxy,sizex, sizey,indices = read_indicses_direction(points, start, end,direction)
        
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
            cv2.circle(img=blank_image, center=(pt[0], pt[1]), radius=0, color=(255, 255, 255), thickness=-1)'''

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
                
                pt1 = [points[index1][0] - minx + margin, points[index1][1] - miny + margin]
                pt2 = [points[index2][0] - minx + margin, points[index2][1] - miny + margin]
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
        cv2.imwrite('gray.jpg',gray)
        xy = out_dict['xy']
        edged = cv2.Canny(gray,30,200)
        cv2.imwrite('edged.jpg',edged)

        # get the countours of the piece
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        
        # find the closes point in the contour to a point
        contour = max(contours, key=cv2.contourArea)
        contour_list = []
        for c in contour:
            # val = c[0].tolist()
            # if val not in contour_list:
            contour_list.append(c[0].tolist())

        

        new_img,new_points = contour_to_image(out_dict,points=contour_list, filename=out_dict['name'])

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

       
# function to check if a point is between two other points
def is_between(p1, p2, p3):
    # check if the point is between the two other points
    # if the point is between the two other points, then the distance between the point and the two other points
    # should be equal to the distance between the two other points
    return distance(p1, p2) + distance(p2, p3) == distance(p1, p3)
