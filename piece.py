# piece dataclass

import os
from os.path import join
import numpy as np
import pandas as pd
import cv2
import math
from dataclasses import dataclass, field
from utils import get_line_through_points, distance_point_line_squared,\
      distance,rotate_points,point_exists,slope_in_degrees,color_to_number
from side_extractor import rotation_matrix
from side import Side   # dataclass
from enum import Enum

# enum for shape of each side
class SideShape(Enum):
    IN = 0
    OUT = 1


@dataclass
class Piece:
    piece_file_name: str = field(init=False, repr=False)
    name: str
    corners: list[list] = field(init=False, repr=False,default_factory=list)
    corners_index: list[int] = field(init=False, repr=False,default_factory=list)
    sides: list[Side] = field(init=False, repr=False,default_factory=list)
    camera_folder: str =  field(init=False, repr=False, default= "camera")
    piece_folder: str = field(init=False, repr=False, default= "pieces")
    threshold_folder: str = field(init=False, repr=False, default= "threshold")
    contour_folder: str = field(init=False, repr=False, default= "contours")
    
    status: str = field(init=False, repr=False, default= "n")
    # edge_points: list[list] = field(init=False, repr=False, default_factory=list)
    # corners: list[list] = field(init=False, repr=False, default_factory=list)
    points_list: list[list] = field(init=False, repr=False, default_factory=list)
    in_out: list[SideShape] = field(init=False, repr=False, default_factory=list)
    # size of the piece
    width: int = field(init=False, repr=False, default= 0)
    height: int = field(init=False, repr=False, default= 0)
    
    def __post_init__(self):
        self.piece_file_name = self.name + ".jpg"
        
        # self.piece_folder_threshold = join(self.piece_folder_threshold, self.name)
        # self.contour_folder = join(self.contour_folder, self.name)
        os.makedirs(self.piece_folder, exist_ok=True)
        os.makedirs(self.threshold_folder, exist_ok=True)
        os.makedirs(self.contour_folder, exist_ok=True)
        self.corners = []
        self.sides = []

    def read_camera_image(self,input_filename: str, stat: str)->bool:
        img = cv2.imread(join("camera", input_filename))
        img = img[1100:1700,1400:2000]
        # img = cv2.GaussianBlur(img,(3,3),0)
        img = cv2.GaussianBlur(img,(5,5),0)
        
        
        ret, thr = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)  
        piece_name = join(self.piece_folder, self.name+".png")
        
        cv2.imwrite(piece_name, img)
        threshold_name = join(self.threshold_folder, self.name+".png")
        cv2.imwrite(threshold_name, thr)

        pixel_matrix = self.get_edge_points(thr)
        self.get_white_pixels(pixel_matrix)
        
        # if(self.contour_to_points(image=thr) is False):
        #     return False    
        if(self.contour_to_image() is False):
            return False
        if(self.find_corner() is False):
            return False
        if(self.order_points_clockwise() is False):
            return False
        if(self.fine_tune_corners() is False):
            return False
        if(self.find_shape_in_out() is False):
            return False
        if(self.shape_classification() is False):
            return False
        return True

    # retunrs a list of points
    def image_to_list(self,image)->list:
        try:
            height, width = image.shape[:2]
            points_list = []
            for y in range(0, height):
                for x in range(0, width):
                    if image[y, x] == 0:
                        points_list.append([x,y])
            return points_list
        except Exception as e:
            print(e)
        

    def contour_to_points(self,image)->bool:
        try:
            gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, gray = cv2.threshold(gray1, 128, 255, cv2.THRESH_BINARY_INV) 
            edged = cv2.Canny(gray,30,200)
            # get the countours of the piece
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, \
                                                cv2.CHAIN_APPROX_TC89_KCOS)
            # find the closes point in the contour to a point
            contour = max(contours, key=cv2.contourArea)
            # contour_list = []
            # for c in contour:
            #     contour_list.append(c[0].tolist())

            self.points_list = [c[0].tolist() for c in contour]
            return True
        except Exception as e:
            print(e)
            return False

    # function to get list of points and maximu size and save the pints in an image
    def contour_to_image(self)-> bool:
        try:
            # blank_image = np.zeros((max_size(0),max_size(1),3), np.uint8)
            # minimum value of x and y
            minx = min(self.points_list, key=lambda x: x[0])[0]
            miny = min(self.points_list, key=lambda x: x[1])[1]
            # maximum value of x and y
            maxx = max(self.points_list, key=lambda x: x[0])[0]
            maxy = max(self.points_list, key=lambda x: x[1])[1]
            margin = 5
            # size of the image
            sizex, sizey = ((maxx - minx + margin*2), (maxy - miny+margin*2))

            blank_image = np.zeros((sizey, sizex, 3), np.uint8)
            self.width, self.height = blank_image.shape[:2]
            # self.edge_points = []
            # draw the contour

            for i in range(len(self.points_list) ):
                self.points_list[i] = [self.points_list[i][0] - minx + margin, \
                        self.points_list[i][1] - miny + margin]
                
            for i in range(len(self.points_list) ):
                index1 = i % len(self.points_list)
                index2 = (i+1) % len(self.points_list)
                
                pt1 = self.points_list[index1]
                pt2 = self.points_list[index2]
                # pt1 = [self.points_list[index1][0] - minx + margin, \
                #         self.points_list[index1][1] - miny + margin]
                # pt2 = [self.points_list[index2][0] - minx + margin, \
                #         self.points_list[index2][1] - miny + margin]
                # self.edge_points.append(pt1)
                cv2.line(blank_image, pt1, pt2, (255, 255, 255), 1)

            ret, gray = cv2.threshold(blank_image, 128, 255, cv2.THRESH_BINARY) 

            # cv2.drawContours(blank_image, new_points, -1, (255,255,255), 1)
            contour_name = join(self.contour_folder, self.name+".png")
            cv2.imwrite(contour_name,gray)
            return True
        
        except Exception as e:
            print(str(e))
            return False

     
    def find_corner(self)->bool:
        try:
            contour_name = join(self.contour_folder, self.name+".png")
            img = cv2.imread(contour_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            corners = cv2.goodFeaturesToTrack(gray, 100, 0.15, 10)
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
                # l = len(corners)
                output_list = []
                count = 0
                for index, row in df.iterrows():
                    x = int(row['X'])
                    y = int(row['Y'])
                    xy = (x,y)
                    
                    color2 = (0,0,255)
                    cv2.circle(img,xy,3,color=color2,thickness=1)
                    cv2.imshow(self.name,img)
                    full_key = cv2.waitKeyEx(0)
                    # print(f"Key pressed {full_key}\n")
                    if full_key == 110:
                        continue
                    if full_key == 120:
                        break
                    count += 1
                    output_list.append(xy)
                    if(count == 4):
                        break
                    
                    
                cv2.destroyWindow(self.name)
                ###############
            self.corners = output_list   
            if(len(self.corners) == 4):
                return True
            else:
                return False
        except Exception as e:
            print(e)
            return False
                                    
    

    def order_points_clockwise(self)->bool:
        try:
            # Initialize the list of ordered points
            # ordered_pts = [None] * 4
            
            # Find the center of the points
            center = [sum(pt[0] for pt in self.corners) // len(self.corners),\
                       sum(pt[1] for pt in self.corners) // len(self.corners)]
            
            # Divide the points into two groups: those above the center and those below
            above_center = []
            below_center = []
            for pt in self.corners:
                if pt[1] < center[1]:
                    above_center.append(pt)
                else:
                    below_center.append(pt)
            
            # Sort the points in each group by their x-coordinate
            above_center = sorted(above_center, key=lambda pt: pt[0])
            below_center = sorted(below_center, key=lambda pt: pt[0], reverse=True)
            
            # Assign the ordered points to the output list
            self.corners[0] = above_center[0]
            self.corners[1] = above_center[-1]
            self.corners[2] = below_center[0]
            self.corners[3] = below_center[-1]
            return True
        except Exception as e:
            print(e)
            return False

    def fine_tune_corners(self)->bool:
        try:
            # find the closest point of coords to x,y
            self.find_nearest_point(0,-1,-1)
            self.find_nearest_point(1,1,-1)
            self.find_nearest_point(2,1,1)
            self.find_nearest_point(3,-1,1)

            # rotate self.points_list n times to get the correct order
            # n = self.corners_index[0]
            # self.points_list = self.points_list[n:] + self.points_list[:n]
            # self.corners_index = [i-n for i in self.corners_index]

            revers_order = True
            for i in range(4):
                if((self.corners_index[i%4] < self.corners_index[(i+1)%4]) and \
                      (self.corners_index[(i+1)%4] < self.corners_index[(i+2)%4])):
                    revers_order = False
            
            temp_list_corners = []
            
            temp_list_piece = []
            no_of_points = len(self.points_list)
            for i in range(4):
                start_index = self.corners_index[i]
                temp_list_corners.append(len(temp_list_piece))
                temp_list_side = []
                # if revers_order:
                #     last_index = self.corners_index[(i-1)%4]
                # else:
                #     last_index = self.corners_index[(i+1)%4]
                last_index = self.corners_index[(i+1)%4]

                while(start_index != last_index):
                    temp_list_side.append(self.points_list[start_index])
                    if revers_order:
                        start_index = (start_index+no_of_points-1)%no_of_points
                    else:
                        start_index = (start_index+1)%no_of_points

                temp_list_piece = temp_list_piece + temp_list_side

                # if(revers_order): 
                #     temp_list_piece = temp_list_piece + temp_list_side[::-1]
                # else:
                #     temp_list_piece = temp_list_piece + temp_list_side

            self.corners_index = temp_list_corners
            self.points_list = temp_list_piece
            file_name = join(self.contour_folder, self.name+".csv")
            f = open(file_name,"w")
            for p in self.points_list:
                f.write(f"{p[0]},{p[1]}\n")
            f.close()
            return True
        except Exception as e:
            print(e)
            return False
        
    
    def find_nearest_point(self, idx, a,b)->bool:
        try:
            """Finds the nearest point in a contour to a given point."""
            point = self.corners[idx]

            # find the point in point_list that is closest to point
            closest_index = find_closest_point_index(self.points_list, point)
            self.corners_index.insert(idx, closest_index)
            x,y = self.points_list[closest_index]
            
            while point_exists(self.points_list, x+a,y+b):
                x = x+a
                y = y+b
            while point_exists(self.points_list, x+a,y):
                x = x+a
            while point_exists(self.points_list, x,y+b):
                y = y+b
            self.corners[idx] = [x,y]



            return True
        except Exception as e:
            print(e)
            return False

    """ def find_nearest_point(self,points, idx, a,b)->bool:
        try:
            #Finds the nearest point in a contour to a given point.
            point = self.corners[idx]
            contour = np.asarray(points)
            dists = np.sqrt(np.sum((contour - point)**2, axis=1))
            nearest_idx = np.argmin(dists)
            nearest_point = contour[nearest_idx]
            x = nearest_point[0]
            y = nearest_point[1]  

            while point_exists(points, x+a,y+b):
                x = x+a
                y = y+b
            while point_exists(points, x+a,y):
                x = x+a
            while point_exists(points, x,y+b):
                y = y+b
            self.corners[idx] = [x,y]
            return True
        except Exception as e:
            print(e)
            return False """

    def find_shape_in_out(self)->bool:
        try:
            # get the center of the xy
            center = [sum(pt[0] for pt in self.corners) // len(self.corners),\
                       sum(pt[1] for pt in self.corners) // len(self.corners)]

            for i in range(4):
                if(i == 3):
                    side_points = self.points_list[self.corners_index[i]:]
                else:
                    side_points = self.points_list[self.corners_index[i]:self.corners_index[(i+1)%4]]
                line = get_line_through_points(self.corners[i],self.corners[(i+1)%4])
                # find the index of maximum distance between points in side_points and line
                ds = [distance_point_line_squared(line, point) for point in side_points]
                max_d = max(ds)
                max_d_index = ds.index(max_d)
                corner1 = self.corners[i]
                corner2 = self.corners[(i+1)%4]    
                pt1 = [(corner1[0]+corner2[0])/2,(corner1[1]+corner2[1])/2]
                d1 = distance(pt1,center)
                d2 = distance(side_points[max_d_index],center)
                if(d1<d2):
                    self.in_out.append(SideShape.OUT)
                else:
                    self.in_out.append(SideShape.IN)
                
            return True
        except Exception as e:
            print(e)
            return False   

    def shape_classification(self)->bool:
        
        try:
            for i in range(4):
                if(i == 3):
                    side_points = self.points_list[self.corners_index[i]:]
                else:
                    side_points = self.points_list[self.corners_index[i]:self.corners_index[(i+1)%4]]
                self.side_to_image(i,side_points)
            return True
        except Exception as e:
            print(e)
            return False

    def draw_points(self,points):
        # blank_image = np.zeros((max_size(0),max_size(1),3), np.uint8)
            # minimum value of x and y
            minx = min(points, key=lambda x: x[0])[0]
            miny = min(points, key=lambda x: x[1])[1]
            # maximum value of x and y
            maxx = max(points, key=lambda x: x[0])[0]
            maxy = max(points, key=lambda x: x[1])[1]

            marg = 1
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

            cv2.imshow('image',blank_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def side_to_image(self, idx: int,points: list)->bool:
        try:
            # show_point(points)
            oriatation = self.in_out[0].value
            
            points = rotate_points(points, \
                                rotation_matrix[oriatation, idx], \
                                    self.corners[idx])
            
            # blank_image = np.zeros((max_size(0),max_size(1),3), np.uint8)
            # minimum value of x and y
            minx = min(points, key=lambda x: x[0])[0]
            miny = min(points, key=lambda x: x[1])[1]
            # maximum value of x and y
            maxx = max(points, key=lambda x: x[0])[0]
            maxy = max(points, key=lambda x: x[1])[1]
            # pt1 is the first point in points
            pt1 = points[0]
            # pt2 is the last point in points
            pt2 = points[-1]
            if(pt1[0] < pt2[0]):
                angle = slope_in_degrees(pt1, pt2)*(-1)
            else:
                angle = slope_in_degrees(pt2, pt1)*(-1)
            if(angle > 90):
                angle =  angle - 180
            else:
                angle = angle 
 
            points = rotate_points(points, \
                                angle, \
                                    self.corners[idx])
            
            # blank_image = np.zeros((max_size(0),max_size(1),3), np.uint8)
            # minimum value of x and y
            minx = min(points, key=lambda x: x[0])[0]
            miny = min(points, key=lambda x: x[1])[1]
            # maximum value of x and y
            maxx = max(points, key=lambda x: x[0])[0]
            maxy = max(points, key=lambda x: x[1])[1]

            marg = 1
            # size of the image
            sizex, sizey = ((maxx - minx + marg*2), (maxy - miny+marg*2))
            blank_image = np.zeros((sizey, sizex, 3), np.uint8)
            # shift all points to (minx,miny) and add margin
            points = [[points[i][0] - minx + marg,points[i][1] - miny + marg] \
                    for i in range(len(points))]
            
            if(points[0][0] > points[-1][0]):
                points = points[::-1]
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
            if self.in_out[idx] == SideShape.IN:
                filename = f"{self.name}_{idx+1}_in"
            else:
                filename = f"{self.name}_{idx+1}_out"
            cv2.imwrite(join('sides', filename +".png"),blank_image)
            
            
            f = open(join('sides', filename +".csv"),"w")
            for p in points:
                f.write(f"{p[0]},{p[1]}\n")
            f.close()

            return True
        except Exception as e:
            print(e)
            return False     

    def get_edge_points(self,img):
        try:
            ret, image = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
            pixel_matrix = image.tolist()
            pixel_matrix = threshold_image(pixel_matrix)
            # Create a copy of the input image
            
            for row in range(1,len(pixel_matrix)-1):
                for col in range(1,len(pixel_matrix[row])-1):
                    if(pixel_matrix[row][col] == [255,255,255]):
                        if(pixel_matrix[row-1][col] == [0,0,0] or \
                           pixel_matrix[row+1][col] == [0,0,0] or \
                           pixel_matrix[row][col-1] == [0,0,0] or \
                           pixel_matrix[row][col+1] == [0,0,0] ):
                            pixel_matrix[row][col] = [128,128,128]

            for row in range(1,len(pixel_matrix)-1):
                for col in range(1,len(pixel_matrix[row])-1):
                    if(pixel_matrix[row][col] == [128,128,128]):
                        pixel_matrix[row][col] = [255,255,255]
                    else:
                        pixel_matrix[row][col] = [0,0,0]
                                       
            # file_name = join(self.contour_folder, self.name+".csv")
            # f = open(file_name,"w")
      
            # for row in range(len(pixel_matrix)):
            #     for col in range(len(pixel_matrix[row])):
            #         if(pixel_matrix[row][col] == [255,255,255]):
            #             f.write(f"{row},{col}\n")
                
            # f.close()
            
            return pixel_matrix
        except Exception as e:
            print(e)

    def get_white_pixels(self,pixel_matrix):
        try:
           
            pixels_list = []
            for row in range(len(pixel_matrix)):
                for col in range(len(pixel_matrix[row])):
                    if(pixel_matrix[row][col] == [255,255,255]):
                        pixels_list.append((row,col))
           
            # create a list of status for each pixel
            status = []
            left_index = []
            right_index = []
            for i in range(len(pixels_list)):
                status.append(0)
                left_index.append(-1)
                right_index.append(-1)
            
            for i in range(len(pixels_list)):
                links = get_adjacent_points(pixels_list,i)
                if(len(links) == 2):
                    status[i] = 1
                    left_index[i] = links[0]
                    right_index[i] = links[1]
                elif(len(links) == 3):
                    p0 = pixels_list[i]
                    p1 = pixels_list[links[0]]
                    p2 = pixels_list[links[1]]
                    p3 = pixels_list[links[2]]
                    if(pixel_distance(p0,p1) == 1) and (pixel_distance(p0,p2) == 1):
                        status[i] = 1
                        left_index[i] = links[0]
                        right_index[i] = links[1]
                    elif(pixel_distance(p0,p2) == 1) and (pixel_distance(p0,p3) == 1):
                        status[i] = 1
                        left_index[i] = links[1]
                        right_index[i] = links[2]
                    elif(pixel_distance(p0,p1) == 1) and (pixel_distance(p0,p3) == 1):
                        status[i] = 1
                        left_index[i] = links[0]
                        right_index[i] = links[2]
                    elif(pixel_distance(p0,p1) == 2) and (pixel_distance(p0,p2) == 2):
                        status[i] = 1
                        left_index[i] = links[2]
                        if(pixel_distance(p1,p3) < pixel_distance(p2,p3)):
                            right_index[i] = links[1]
                        else:
                            right_index[i] = links[0]
                    elif(pixel_distance(p0,p1) == 2) and (pixel_distance(p0,p3) == 2):
                        status[i] = 1
                        left_index[i] = links[1]
                        if(pixel_distance(p1,p2) < pixel_distance(p3,p2)):
                            right_index[i] = links[2]
                        else:
                            right_index[i] = links[0]
                    elif(pixel_distance(p0,p2) == 2) and (pixel_distance(p0,p3) == 2):
                        status[i] = 1
                        left_index[i] = links[0]
                        if(pixel_distance(p2,p1) < pixel_distance(p3,p1)):
                            right_index[i] = links[2]
                        else:
                            right_index[i] = links[1]
                    else:
                        print(f"error: {i} {links} {pixels_list[i]}")  
                        return False
                    
                else:
                    print(f"error: {i} {links} {pixels_list[i]}")
                    return False
            
            # Return the list of white pixels
            for i in range(len(pixels_list)):
                if(status[i] == 0):
                    print(f"Error in  index {i}")
                    return False
            
            start_point = 0
            next_point = left_index[0]
            self.points_list.append(pixels_list[start_point])
           
            while(next_point != 0):
                self.points_list.append(pixels_list[next_point])
                new_point = next_pixel(next_point,start_point,left_index,right_index)
                start_point = next_point
                next_point = new_point
                
            


            return True
        except Exception as e:
            print(e)
            return False

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
            if pixel_matrix[row][col] != [255, 255, 255]:
                pixel_matrix[row][col] = [0,0,0]
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
