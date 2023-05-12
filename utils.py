import numpy as np
import cv2
import math

def distance_point_line_squared(a_b_c : tuple, x0_y0 : tuple):
    """
    Computes the squared distance of a 2D point (x0, y0) from a line ax + by + c = 0
    """
    a ,b ,c = a_b_c
    x0 ,y0 = x0_y0
    return abs(a*x0 + b*y0 + c) / (a**2 + b**2)**0.5



# function to return distance between two points
def distance(x1_y1, x2_y2):
    """
    Computes the distance between two points (x1, y1) and (x2, y2)
    """
    x1, y1 = x1_y1
    x2, y2 = x2_y2
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def get_line_through_points(p0, p1):
    """
    Given two points p0 (x0, y0) and p1 (x1, y1),
    compute the coefficients (a, b, c) of the line 
    that passes through both points.
    """
    x0, y0 = p0
    x1, y1 = p1
    
    return y1 - y0, x0 - x1, x1*y0 - x0*y1




def distance_point_line_signed(a_b_c : tuple, x0_y0 : tuple):
    """
    Computes the signed distance of a 2D point (x0, y0) from a line ax + by + c = 0
    """
    a ,b ,c = a_b_c
    x0 ,y0 = x0_y0
    
    return (a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)


def rotate(image, degrees):
    """
    Rotate an image by the amount specifiedi in degrees
    """
    if len(image.shape) == 3:
        rows,cols, _ = image.shape
    else:
        rows, cols = image.shape
        
    M = cv2.getRotationMatrix2D((cols/2,rows/2), degrees, 1)
    
    return cv2.warpAffine(image,M,(cols,rows)), M

def intersection_point(line1_point1, line1_point2, line2_point1, line2_point2):
    # extract x and y coordinates of the four points
    x1, y1 = line1_point1
    x2, y2 = line1_point2
    x3, y3 = line2_point1
    x4, y4 = line2_point2
    
    # calculate the slopes and y-intercepts of each line
    slope1 = (y2 - y1) / (x2 - x1)
    y_int1 = y1 - slope1 * x1
    
    slope2 = (y4 - y3) / (x4 - x3)
    y_int2 = y3 - slope2 * x3
    
    # calculate the intersection point
    x_int = (y_int2 - y_int1) / (slope1 - slope2)
    y_int = slope1 * x_int + y_int1
    
    return (x_int, y_int)

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
    # result = [[int(round(x) + px), int(round(y) + py)] for x, y in rotated]
    result = [[int(x + px), int(y + py)] for x, y in rotated]


    return result

def point_exists(points, x, y):
    """
    Check if a [x,y] point exists in the list of points.
    Args:
        points (list): a list of [x,y] points
        x (int): the x coordinate of the point to search for
        y (int): the y coordinate of the point to search for
    Returns:
        bool: True if the point exists in the list, False otherwise
    """
    for point in points:
        if point[0] == x and point[1] == y:
            return True
    return False


def slope_in_degrees(point1, point2):
    # Calculate the slope of the line in radians
    slope_in_radians = math.atan2(point2[1] - point1[1], point2[0] - point1[0])
    
    # Convert the slope from radians to degrees
    slope_in_degrees = math.degrees(slope_in_radians)
    
    return slope_in_degrees

# return pixel color as a number
def color_to_number(pixel_color):
    """
    Get the pixel color at a given x and y coordinate
    """
    return (pixel_color[0] << 16) + (pixel_color[1] << 8) + pixel_color[2]


# function to rotate an image
def rotate_image(image, angle):
    # get the image size
    image_size = (image.shape[1], image.shape[0])
    # get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((image_size[0] / 2, image_size[1] / 2), angle, 1.0)
    # rotate the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, image_size, flags=cv2.INTER_LINEAR)
     # Convert the rotated image to grayscale
    gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to convert the grayscale image to black and white
    _, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Return the black and white image
    return bw_image



def next_pixel(next_point,start_point,left_index,right_index):
    # get the next pixel in the contour
    if(start_point == left_index[next_point]):
        return right_index[next_point]
    else:
        return left_index[next_point]
    

# check if the points are diagonal
def pixel_distance(p1,p2)->int:
    # distance between the points
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        

def get_adjacent_points(white_pixels_list,idx):
    # get the adjacent points of the idx-th point in the list
    x = white_pixels_list[idx][0]
    y = white_pixels_list[idx][1]
    adjacent_list = []
    for i in range(-1,2):
        for j in range(-1,2):
            if(i != 0 or j != 0):
                ind = search_pixel_list(white_pixels_list,x+i,y+j)
                if(ind != -1):
                    adjacent_list.append(ind)
    return adjacent_list

def search_pixel_list(white_pixels_list,x,y):
    # search for the pixel in the list
    # if found return the index
    # otherwise return -1
    for i in range(len(white_pixels_list)):
        if(white_pixels_list[i][0] == x and white_pixels_list[i][1] == y):
            return i
    return -1


def is_pixel_and_surrounding_white(pixel_matrix, row, col):
    # Check if the pixel is white
    if pixel_matrix[row][col] != [255, 255, 255]:
        return False
    # Check if all surrounding pixels are white
    for i in range(row-1, row+2):
        for j in range(col-1, col+2):
            if pixel_matrix[i][j] != [255, 255, 255]:
                return False
    return True

def threshold_image(pixel_matrix):
    pixel_matrix = transpose_matrix(pixel_matrix)
    # Iterate over all pixels in the matrix
    for row in range(len(pixel_matrix)):
        for col in range(len(pixel_matrix[row])):
            # If the pixel is not white, set it to black
            if pixel_matrix[row][col] != 255:
                pixel_matrix[row][col] = 0
            # else:
            #     pixel_matrix[row][col] = 0xffffff
    return pixel_matrix

def transpose_matrix(pixel_matrix):
    # Use the zip function to transpose the matrix
    transposed_matrix = list(map(list, zip(*pixel_matrix)))
    return transposed_matrix





def find_closest_point_index(points, target):
    closest_index = None
    closest_distance = float('inf')
    for i, point in enumerate(points):
        distance = math.sqrt((point[0]-target[0])**2 + (point[1]-target[1])**2)
        if distance < closest_distance:
            closest_index = i
            closest_distance = distance
    return closest_index


def make_black_transparent(image):
    # Convert image to RGBA
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    
    # Create a mask for the black pixels
    black = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    black[np.where((image[:,:,0] == 0) & (image[:,:,1] == 0) & (image[:,:,2] == 0))] = 255
    
    # Set the alpha channel to the mask
    rgba[:,:,3] = black
    
    # Convert back to BGR
    result = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    
    return result




def moravec_corner_detection(img, window_size=3, threshold=500):
    # Compute image gradients in x and y directions
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the sum of squared differences in all directions
    h, w = img.shape
    m = np.zeros((h, w))
    for y in range(window_size//2, h-window_size//2):
        for x in range(window_size//2, w-window_size//2):
            min_ssd = float('inf')
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    ssd = np.sum((img[y:y+window_size, x:x+window_size] - img[y+dy:y+dy+window_size, x+dx:x+dx+window_size])**2)
                    if ssd < min_ssd:
                        min_ssd = ssd
            m[y, x] = min_ssd
    
    # Threshold the corner response
    corners = np.zeros((h, w), np.uint8)
    corners[m > threshold] = 255
    
    return corners



def moravec_corner_points(img, window_size=3, threshold=500):
    h, w = img.shape
    corners = np.zeros((h, w))
    for y in range(window_size//2, h-window_size//2):
        for x in range(window_size//2, w-window_size//2):
            win = img[y-window_size//2:y+window_size//2+1, x-window_size//2:x+window_size//2+1]
            min_diff = np.inf
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    shifted_win = img[y+dy-window_size//2:y+dy+window_size//2+1, x+dx-window_size//2:x+dx+window_size//2+1]
                    diff = np.sum((win - shifted_win)**2)
                    if diff < min_diff:
                        min_diff = diff
            if min_diff > threshold:
                corners[y, x] = 1
    corner_points = np.transpose(np.nonzero(corners))
    return corner_points


def harris_corner_detection(img, k=0.04, window_size=3, threshold=0.01):
    # Compute image gradients in x and y directions
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute products of gradients
    Ixx = dx**2
    Ixy = dx*dy
    Iyy = dy**2

    # Compute sums of products of gradients within a window
    kernel = np.ones((window_size, window_size), np.float32) / (window_size**2)
    Sxx = cv2.filter2D(Ixx, -1, kernel)
    Sxy = cv2.filter2D(Ixy, -1, kernel)
    Syy = cv2.filter2D(Iyy, -1, kernel)

    # Compute the Harris corner response
    det = Sxx*Syy - Sxy**2
    trace = Sxx + Syy
    R = det - k*trace**2

    # Threshold the corner response
    corners = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    corners[R > threshold*np.max(R)] = 255

    return corners



def harris_corner_points(img, k=0.04, window_size=3, threshold=0.01):
    # Compute image gradients in x and y directions
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute products of gradients at each pixel
    Ixx = dx**2
    Ixy = dx*dy
    Iyy = dy**2
    
    # Compute the sums of products of gradients at each pixel over a local window
    ksize = (window_size, window_size)
    Sxx = cv2.boxFilter(Ixx, -1, ksize, normalize=False)
    Sxy = cv2.boxFilter(Ixy, -1, ksize, normalize=False)
    Syy = cv2.boxFilter(Iyy, -1, ksize, normalize=False)
    
    # Compute the corner response function at each pixel
    det = Sxx*Syy - Sxy**2
    trace = Sxx + Syy
    response = det - k*trace**2
    
    # Threshold the corner response and compute local maxima
    corners = np.zeros_like(img)
    response[response < threshold*np.max(response)] = 0
    local_maxima = cv2.dilate(response, None)
    corners[local_maxima == response] = 255
    
    # Find the coordinates of the corner points
    corner_points = np.transpose(np.nonzero(corners))
    
    return corner_points


def shi_tomasi_corner_detection(img, max_corners=23, quality_level=0.01, min_distance=10):
    # Compute image gradients in x and y directions
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute products of gradients
    Ixx = dx**2
    Ixy = dx*dy
    Iyy = dy**2

    # Compute sums of products of gradients within a window
    kernel = np.ones((3, 3), np.float32) / 9
    Sxx = cv2.filter2D(Ixx, -1, kernel)
    Sxy = cv2.filter2D(Ixy, -1, kernel)
    Syy = cv2.filter2D(Iyy, -1, kernel)

    # Compute the Shi-Tomasi corner response
    R = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            M = np.array([[Sxx[y, x], Sxy[y, x]], [Sxy[y, x], Syy[y, x]]])
            eigenvalues = np.linalg.eigvals(M)
            R[y, x] = np.min(eigenvalues)

    # Find the strongest corners
    corners = cv2.goodFeaturesToTrack(R, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)

    # Draw circles around the corners
    for corner in corners:
        x, y = corner[0]
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

    return img



# Shi-Tomasi corner detection
def shi_tomasi_corner_points(img, max_corners=100, quality_level=0.01, min_distance=10):
    # Compute the eigenvalues of the structure tensor at each pixel
    eigvals = cv2.cornerEigenValsAndVecs(img, blockSize=3, ksize=3)
    eigvals = eigvals.reshape(img.shape[0], img.shape[1], 3, 2)
    eigvals = np.sort(eigvals, axis=2)
    # Compute the minimum eigenvalue at each pixel
    min_eigvals = eigvals[:, :, 0]
    # Compute the corner response function at each pixel
    response = np.zeros_like(img)
    response = cv2.normalize(min_eigvals, response, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Find the coordinates of the local maxima in the corner response function
    corner_points = cv2.goodFeaturesToTrack(response, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    corner_points = corner_points.reshape(-1, 2)
    return corner_points



def find_line(points_list, point_index):
    # Extract the current point and the next four points
    current_point = points_list[point_index]
    next_points = points_list[point_index+1:point_index+5]
    
    # Create an array of x-values and y-values for the points
    x_values = np.array([point[0] for point in next_points])
    y_values = np.array([point[1] for point in next_points])
    
    # Calculate the least-squares line of best fit through the points
    A = np.vstack([x_values, np.ones(len(x_values))]).T
    m, c = np.linalg.lstsq(A, y_values, rcond=None)[0]
    
    # Return the slope and y-intercept of the line
    return (m, c)

def find_lines_and_angle(points_list, point_index):
    # Extract the current point and the next/previous four points
    list_size = len(points_list)
    next_points = []
    prev_points = []
    for i in range(4):
        next_points.append(points_list[(point_index + i) % list_size])
        prev_points.append(points_list[(list_size + point_index - i) % list_size])

    # Create an array of x-values and y-values for the next/previous points
    x_next = np.array([point[0] for point in next_points])
    y_next = np.array([point[1] for point in next_points])
    x_prev = np.array([point[0] for point in prev_points])
    y_prev = np.array([point[1] for point in prev_points])
    
    # Calculate the least-squares line of best fit through the next/previous points
    A_next = np.vstack([x_next, np.ones(len(x_next))]).T
    m_next, c_next = np.linalg.lstsq(A_next, y_next, rcond=None)[0]
    A_prev = np.vstack([x_prev, np.ones(len(x_prev))]).T
    m_prev, c_prev = np.linalg.lstsq(A_prev, y_prev, rcond=None)[0]
    
    # Calculate the angle between the lines
    angle = np.arctan((m_next - m_prev) / (1 + m_next * m_prev))

    # convert to dgree
    angle = angle * 180 / np.pi
    
    # get absoulte value
    angle = abs(angle)

    # Return the slopes, y-intercepts, and angle of the lines
    # return ((m_prev, c_prev), (m_next, c_next), angle)
    return angle

    


def find_lines_interpolate_and_angle(points_list, point_index):
    # Calculate the start and end indices for the previous and next points
    list_size = len(points_list)
    next_points = []
    prev_points = []
    for i in range(10):
        next_points.append(points_list[(point_index + i) % list_size])
        prev_points.append(points_list[(list_size + point_index - i) % list_size])

    
    # Create an array of x-values and y-values for the next/previous points
    x_prev = np.array([point[0] for point in prev_points])
    y_prev = np.array([point[1] for point in prev_points])
    x_next = np.array([point[0] for point in next_points])
    y_next = np.array([point[1] for point in next_points])
    
    # Interpolate linearly between the next/previous points
    m_prev, c_prev = np.polyfit(x_prev, y_prev, 1)
    m_next, c_next = np.polyfit(x_next, y_next, 1)
    
    # Calculate the angle between the lines
    angle = np.arctan((m_next - m_prev) / (1 + m_next * m_prev))
    # convert to dgree
    angle = angle * 180 / np.pi
    
    # get absoulte value
    angle = abs(angle)
    # Return the slopes, y-intercepts, and angle of the lines
    # return ((m_prev, c_prev), (m_next, c_next), angle)
    return angle

def find_lines_interpolate(points_list):
    
    
    # Create an array of x-values and y-values for the next/previous points
    x_list = np.array([point[0] for point in points_list])
    y_list = np.array([point[1] for point in points_list])

    # Interpolate linearly between the next/previous points
    m, c = np.polyfit(x_list, y_list, 1)
    
    return (m, c)

def find_intersection(x, y, a, b, c):
    if(b == 0):
        return -(c/a), y
    if(a == 0):
        return x,-(c/b)

    slope = -(a/b)
    start = -(c/b)
    slope_perpendicular = -1 / slope
    start_perpendicular = y - (slope_perpendicular * x)
    intersection_x = (start_perpendicular - start) / (slope - slope_perpendicular)
    intersection_y = slope * intersection_x + start
    return intersection_x, intersection_y
    
