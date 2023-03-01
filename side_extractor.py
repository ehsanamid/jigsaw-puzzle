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
from utils import get_line_through_points, distance_point_line_squared, distance_point_line_signed, rotate

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


def shape_classification(out_dict,img,edges, line_params):
    
    # First part: we take all edge points and classify them only if their distance to one of the 4 piece
    # lines is smaller than a certain threshold. If that happens, we can be certain that the point belongs
    # to that side of the piece. If each one of the four distances is higher than the threshold, the point
    # will be classified during the second phase.


    class_image = np.zeros(img.shape, dtype='uint8')
    non_classified_points = []

    blank_image = []
    blank_image.append(np.zeros(img.shape, np.uint8))
    blank_image.append(np.zeros(img.shape, np.uint8))
    blank_image.append(np.zeros(img.shape, np.uint8))
    blank_image.append(np.zeros(img.shape, np.uint8))
    filename = out_dict['name'] 

    for _edge in edges:
        d = [distance_point_line_squared(line_param, _edge) for line_param in line_params]
        ind = np.argmin(d)
        cv2.circle(img=blank_image[ind], center=_edge, radius=0, color=(255,255,255), thickness=-1)
        cv2.imshow(filename+"_1",blank_image[0])
        cv2.imshow(filename+"_2",blank_image[1])
        cv2.imshow(filename+"_3",blank_image[2])
        cv2.imshow(filename+"_4",blank_image[3])
               
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    
    for i in range(4):
        cv2.imwrite(join('sides', filename+"_"+str(i+1)+".jpg"), blank_image[i])  

            
    return class_image


def compute_inout(class_image, line_params, xb_yb, d_threshold=10):
    
    # Given the full class image, the line parameters and the coordinates of the barycenter,
    # compute for each side if the curve of the piece goes inside (in) or outside (out).
    # This is done by computing the mean coordinates for each class and see if the signed distance
    # from the corners' line has the same sign of the signed distance of the barycenter. If that
    # true, the two points lie on the same side and we have a in; otherwise we have a out.
    # To let the points of the curve to contribute more to the mean point calculation, only the
    # signed distances that are greater than a threshold are used.
    
    inout = []
    xb = xb_yb[0]
    yb = xb_yb[1]
    for line_param, cl in zip(line_params, (1, 2, 3, 4)):

        coords = np.array([zip(*np.where(class_image == cl))])[0]

        distances = np.array([distance_point_line_signed(line_param, (x0, y0)) for y0, x0 in coords])    
        distances = distances[np.abs(distances) > d_threshold]
        m_dist = np.mean(distances)

        b_dist = distance_point_line_signed(line_param, (xb, yb))

        if b_dist * m_dist > 0:
            inout.append('in')
        else:
            inout.append('out')
            
    return inout


def create_side_images(class_image, inout, corners):
    
    how_to_rotate = [(90, -90), (180, 0), (-90, 90), (0, 180)]
    side_images = []

    for cl in (1, 2, 3, 4):

        side_image = np.zeros(class_image.shape, dtype='uint8')
        side_image[class_image == cl] = cl

        io = inout[cl - 1]
        htw = how_to_rotate[cl - 1]
        side_corners_idx = _corner_indexes[cl - 1]

        htw = htw[0] if io == 'in' else htw[1]
        # side_image_rot, M = rotate(side_image, htw)
        side_image_rot = side_image

        # side_corners = np.array(np.round([M.dot((corners[corner_idx][0], corners[corner_idx][1], 1)) 
        #                                   for corner_idx in side_corners_idx])).astype(np.int)

        # Order the corners from higher (smaller y coordinate)
        # if side_corners[0, 1] > side_corners[1, 1]:
        #     side_corners = side_corners[::-1]

            
        # Correct the angle on each side separately
        # if side_corners[0, 0] != side_corners[1, 0]:
        #     m = float(side_corners[1, 1] - side_corners[0, 1]) / (side_corners[1, 0] - side_corners[0, 0])
        #     corners_angle = np.arctan(m) * 180 / np.pi
        #     correction_angle = - (corners_angle / abs(corners_angle) * 90 - corners_angle)

        #     side_image_rot, M = rotate(side_image_rot, correction_angle)

        side_image_rot[side_image_rot <= 0.5] = 0
        side_image_rot[side_image_rot > 0.5] = 1
        
        
        nz = np.nonzero(side_image_rot)
        min_y, max_y, min_x, max_x = np.min(nz[0]), np.max(nz[0]), np.min(nz[1]), np.max(nz[1])
        side_image_rot = side_image_rot[min_y:max_y+1, min_x:max_x+1]

        side_images.append(side_image_rot)
            
    return side_images


def plot_side_images(side_images, inout):
    
    for cl, (side_image, io) in enumerate(zip(side_images, inout), start=1):

        plt.subplot(220 + cl)
        plt.title("{0} {1}".format(cl, io))
        plt.imshow(cv2.dilate(side_image, (3,3)))


def draw_lines(image, lines, color):

    for rho, theta in lines:

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(image, (x1,y1), (x2,y2), color, 2)

    return image


def cluster_lines(lines):
    
    # split based on angle
    sines = np.sin(lines[:, 1])
    ordinates = np.abs(lines[:, 0])
    
    kmeans_angle = KMeans(n_clusters=2).fit(sines.reshape(-1, 1))
    ord0 = ordinates[kmeans_angle.labels_ == 0]
    ord1 = ordinates[kmeans_angle.labels_ == 1]
    
    print(sines[kmeans_angle.labels_ == 0])
    print(sines[kmeans_angle.labels_ == 1])
    
    # split based on ordinate
    kmeans_lines0 = KMeans(n_clusters=2).fit(ord0.reshape(-1, 1))
    kmeans_lines1 = KMeans(n_clusters=2).fit(ord1.reshape(-1, 1))
    
    count_lines0 = 0
    count_lines1 = 0
    
    final_labels = []
    for idx in range(len(lines)):
        angle_label = kmeans_angle.labels_[idx]
        if angle_label == 0:
            coord_label = kmeans_lines0.labels_[count_lines0]
            count_lines0 += 1
            
            if coord_label == 0:
                final_labels.append(0)
            else:
                final_labels.append(1)
        else:
            coord_label = kmeans_lines1.labels_[count_lines1]
            count_lines1 += 1
            
            if coord_label == 0:
                final_labels.append(2)
            else:
                final_labels.append(3)
                
    return np.array(final_labels)



        
def get_corners(dst, neighborhood_size=5, score_threshold=0.3, minmax_threshold=100)-> np.array:
    
    """
    Given the input Harris image (where in each pixel the Harris function is computed),
    extract discrete corners
    """
    data = dst.copy()
    data[data < score_threshold*dst.max()] = 0.

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > minmax_threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    yx = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
    return yx[:, ::-1]

    

def get_best_fitting_rect_coords(xy, d_threshold=30, perp_angle_thresh=20, verbose=0):

    """
    Since we expect the 4 puzzle corners to be the corners of a rectangle, here we take
    all detected Harris corners and we find the best corresponding rectangle.
    We perform a recursive search with max depth = 2:
    - At depth 0 we take one of the input point as the first corner of the rectangle
    - At depth 1 we select another input point (with distance from the first point greater
        then d_threshold) as the second point
    - At depth 2 and 3 we take the other points. However, the lines 01-12 and 12-23 should be
        as perpendicular as possible. If the angle formed by these lines is too much far from the
        right angle, we discard the choice.
    - At depth 3, if a valid candidate (4 points that form an almost perpendicular rectangle) is found,
        we add it to the list of candidates.
        
    Given a list of candidate rectangles, we then select the best one by taking the candidate that maximizes
    the function: area * Gaussian(rectangularness)
    - area: it is the area of the candidate shape. We expect that the puzzle corners will form the maximum area
    - rectangularness: it is the mse of the candidate shape's angles compared to a 90 degree angles. The smaller
                        this value, the most the shape is similar toa rectangle.
    """
    N = len(xy)

    distances = scipy.spatial.distance.cdist(xy, xy)
    distances[distances < d_threshold] = 0

    def compute_angles(xy):

        angles = np.zeros((N, N))

        for i in range(N):
            for j in range(i + 1, N):

                point_i, point_j = xy[i], xy[j]
                if point_i[0] == point_j[0]:
                    angle = 90
                else:
                    angle = np.arctan2(point_j[1] - point_i[1], point_j[0] - point_i[0]) * 180 / np.pi

                angles[i, j] = angle
                angles[j, i] = angle

        return angles

    angles = compute_angles(xy)
    possible_rectangles = []

    def search_for_possible_rectangle(idx, prev_points=[]):

        curr_point = xy[idx]
        depth = len(prev_points)

        if depth == 0:
            right_points_idx = np.nonzero(np.logical_and(xy[:, 0] > curr_point[0], distances[idx] > 0))[0]
            
            if verbose >= 2:
                print ('point', idx, curr_point)
                
            for right_point_idx in right_points_idx:
                search_for_possible_rectangle(right_point_idx, [idx])

            if verbose >= 2:
                print()
                
            return


        last_angle = angles[idx, prev_points[-1]]
        perp_angle = last_angle - 90
        if perp_angle < 0:
            perp_angle += 180

        if depth in (1, 2):

            if verbose >= 2:
                print ('\t' * depth, 'point', idx, '- last angle', last_angle, '- perp angle', perp_angle)

            diff0 = np.abs(angles[idx] - perp_angle) <= perp_angle_thresh
            diff180_0 = np.abs(angles[idx] - (perp_angle + 180)) <= perp_angle_thresh
            diff180_1 = np.abs(angles[idx] - (perp_angle - 180)) <= perp_angle_thresh
            all_diffs = np.logical_or(diff0, np.logical_or(diff180_0, diff180_1))
            
            diff_to_explore = np.nonzero(np.logical_and(all_diffs, distances[idx] > 0))[0]

            if verbose >= 2:
                print ('\t' * depth, 'diff0:', np.nonzero(diff0)[0], 'diff180:', np.nonzero(diff180)[0], 'diff_to_explore:', diff_to_explore)

            for dte_idx in diff_to_explore:
                if dte_idx not in prev_points: # unlickly to happen but just to be certain
                    next_points = prev_points[::]
                    next_points.append(idx)

                    search_for_possible_rectangle(dte_idx, next_points)
                
        if depth == 3:
            angle41 = angles[idx, prev_points[0]]

            diff0 = np.abs(angle41 - perp_angle) <= perp_angle_thresh
            diff180_0 = np.abs(angle41 - (perp_angle + 180)) <= perp_angle_thresh
            diff180_1 = np.abs(angle41 - (perp_angle - 180)) <= perp_angle_thresh
            dist = distances[idx, prev_points[0]] > 0

            if dist and (diff0 or diff180_0 or diff180_1):
                rect_points = prev_points[::]
                rect_points.append(idx)
                
                if verbose == 2:
                    print ('We have a rectangle:', rect_points)

                already_present = False
                for possible_rectangle in possible_rectangles:
                    if set(possible_rectangle) == set(rect_points):
                        already_present = True
                        break

                if not already_present:
                    possible_rectangles.append(rect_points)

    if verbose >= 2:
        print('Coords')
        print(xy)
        print()
        print('Distances')
        print(distances)
        print()
        print('Angles')
        print(angles)
        print()
    
    for i in range(N):
        search_for_possible_rectangle(i)                 
                    
    if len(possible_rectangles) == 0:
        return None

    def PolyArea(x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    areas = []
    rectangularness = []
    diff_angles = []

    for r in possible_rectangles:
        points = xy[r]
        areas.append(PolyArea(points[:, 0], points[:, 1]))

        mse = 0
        da = []
        for i1, i2, i3 in [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]:
            diff_angle = abs(angles[r[i1], r[i2]] - angles[r[i2], r[i3]])
            da.append(abs(diff_angle - 90))
            mse += (diff_angle - 90) ** 2

        diff_angles.append(da)
        rectangularness.append(mse)


    areas = np.array(areas)
    rectangularness = np.array(rectangularness)

    scores = areas * scipy.stats.norm(0, 150).pdf(rectangularness)
    best_fitting_idxs = possible_rectangles[np.argmax(scores)]
    return xy[best_fitting_idxs]
        

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

def read_indicses_direction(points, start, end,direction:int):
    try:
        """Reads the indices of a contour in a given direction."""
        
        buffer_size=len(points)

        start_idx = np.argmin(np.sum((points - start)**2, axis=1))
        end_idx = np.argmin(np.sum((points - end)**2, axis=1))
        
        # Calculate the indices of the points between p1 and p2
        if(end_idx < start_idx):
            start_idx, end_idx = end_idx, start_idx

        indices1 = np.arange(start_idx, end_idx + 1)
        indices2 = np.concatenate((np.arange(end_idx, buffer_size), np.arange(start_idx + 1)))

        minx1,miny1,maxx1,maxy1 = get_max_min_x_y(points,indices1)
        minx2,miny2,maxx2,maxy2 = get_max_min_x_y(points,indices2)

        sizex1, sizey1 = get_side_size(minx1,miny1,maxx1,maxy1) 
        sizex2, sizey2 = get_side_size(minx2,miny2,maxx2,maxy2)

        if(sizex1 * sizey1 > sizex2 * sizey2):
            return minx2, miny2, maxx2, maxy2,sizex2,sizey2,indices2
        else:
            return minx1, miny1, maxx1, maxy1,sizex1,sizey1,indices1
    except Exception as e:
        print(str(e))

def circular_buffer(points, filename: str, count:int, start, end,direction:int=0):
    try:
        minx, miny, maxx, maxy,sizex, sizey,indices = read_indicses_direction(points, start, end,direction)
        
        # Create a new image with a black background
        image = np.zeros((sizey, sizex, 3), dtype=np.uint8)

        # Draw lines between the selected points on the image
        for i in range(len(indices) - 1):
            pt1 = (points[indices[i]][0] - minx, points[indices[i]][1] - miny)
            pt2 = (points[indices[i+1]][0] - minx, points[indices[i+1]][1] - miny)
            
            cv2.line(image, pt1, pt2, (255, 255, 255), 1)

        # Save the image to disk
        cv2.imwrite(join('sides', filename+"_"+str(count)+".jpg"), image)
    except Exception as e:
        print(str(e))

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
        marg = 10
        # size of the image
        sizex, sizey = ((maxx - minx + marg*2), (maxy - miny+marg*2))

        blank_image = np.zeros((sizey, sizex, 3), np.uint8)

         

        '''for pt1 in points: 
            pt = [pt1[0] - minx + marg, pt1[1] - miny + marg]
            cv2.circle(img=blank_image, center=(pt[0], pt[1]), radius=0, color=(255, 255, 255), thickness=-1)'''

        for i in range(len(out_dict['xy'])):
            out_dict['xy'][i] = [out_dict['xy'][i][0] - minx + marg, out_dict['xy'][i][1] - miny + marg]
        
        lines = []

        
        lines.append(get_line_through_points(out_dict['xy'][0],out_dict['xy'][1]))
        lines.append(get_line_through_points(out_dict['xy'][1],out_dict['xy'][2]))
        lines.append(get_line_through_points(out_dict['xy'][2],out_dict['xy'][3]))
        lines.append(get_line_through_points(out_dict['xy'][3],out_dict['xy'][0]))

        new_points = []
        # draw the contour
        for i in range(len(points) ):
            index1 = i % len(points)
            index2 = (i+1) % len(points)
            
            pt1 = [points[index1][0] - minx + marg, points[index1][1] - miny + marg]
            pt2 = [points[index2][0] - minx + marg, points[index2][1] - miny + marg]
            new_points.append(pt1)
            cv2.line(blank_image, pt1, pt2, (255, 255, 255), 1)

        ret, gray = cv2.threshold(blank_image, 128, 255, cv2.THRESH_BINARY) 

        # cv2.drawContours(blank_image, new_points, -1, (255,255,255), 1)
        cv2.imwrite(join('contours', filename +".jpg"),gray)
        return blank_image,lines,new_points
    except Exception as e:
        print(str(e))




def process_piece1(image,out_dict, **kwargs):
    
    params = get_default_params()
    for key in kwargs:
        params[key] = kwargs[key]
    
    # out_dict = {}
    
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

        

        new_img,lines,new_points = contour_to_image(out_dict,points=contour_list, filename=out_dict['name'])

        shape_classification(out_dict,new_img,new_points,lines)


        
        
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

       

