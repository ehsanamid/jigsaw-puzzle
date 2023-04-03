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
    result = [[int(round(x) + px), int(round(y) + py)] for x, y in rotated]


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