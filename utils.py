import numpy as np
import cv2

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