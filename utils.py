import numpy as np
import cv2

def distance_point_line_squared(a_b_c : tuple, x0_y0 : tuple):
    """
    Computes the squared distance of a 2D point (x0, y0) from a line ax + by + c = 0
    """
    a = a_b_c[0]
    b = a_b_c[1]
    c = a_b_c[2]
    x0 = x0_y0[0]
    y0 = x0_y0[1]

    return (a*x0 + b*y0 + c)**2 / (a**2 + b**2)

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
    a = a_b_c[0]
    b = a_b_c[1]
    c = a_b_c[2]
    x0 = x0_y0[0]
    y0 = x0_y0[1]
    
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