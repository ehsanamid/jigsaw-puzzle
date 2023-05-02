# piece dataclass

import os
from os.path import join
import numpy as np
import pandas as pd
import cv2
import math
from dataclasses import dataclass, field
from utils import get_line_through_points, distance_point_line_squared,\
      distance,rotate_points,point_exists,slope_in_degrees
from side_extractor import rotation_matrix
from side import Side   # dataclass
from enum import Enum

# enum for shape of each side
class SideShape(Enum):
    UNDEFINED = 0
    IN = 1
    OUT = 2
    

# enum for shape of each side
class ShapeStatus(Enum):
    Piece = 0
    Edge = 1
    Side = 2


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
        os.makedirs(self.piece_folder, exist_ok=True)
        os.makedirs(self.threshold_folder, exist_ok=True)
        os.makedirs(self.contour_folder, exist_ok=True)
        self.corners = []
        self.sides = []


    def camera_image_to_piece(self,input_filename: str,\
                              folder_name: str, stat: str)->bool:
        try:
            piece_name = join(self.piece_folder, self.name+".png")
            threshold_name = join(self.threshold_folder, self.name+".png")
            img = cv2.imread(join(folder_name, input_filename))

            # 
            # img = img[2000:2600,1300:1900]
            img = img[1400:2000,1050:1650]
            # img = cv2.GaussianBlur(img,(3,3),0)
            img = cv2.GaussianBlur(img,(7,7),0)
            # cv2.imshow("img1",img)
            # cv2.waitKey(0)

            # flip the img horizontally and vertically
            img = cv2.flip(img, -1)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            
            # img = cv2.bilateralFilter(img, 9, 75, 75)

            kernel_3x3 = np.ones((3,3),np.float32)/9
            kernel_7x7 = np.ones((7,7),np.float32)/49
            kernel_9x9 = np.ones((9,9),np.float32)/81

            """ kernel_sharpening = np.array([[-1,-1,-1],\
                                            [-1, 9,-1],\
                                            [-1,-1,-1]]) """

            gray1 = cv2.filter2D(gray,-1,kernel_9x9)
            # img = cv2.filter2D(img,-1,kernel_sharpening)
            # cv2.imshow("img2",img1)
            # cv2.waitKey(0)
                
            ret, thr = cv2.threshold(gray1, 120, 255, cv2.THRESH_BINARY)  
            # cv2.imshow(self.name,thr)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            cv2.imwrite(piece_name, img)
            cv2.imwrite(threshold_name, thr)

            # pixel_matrix = self.get_edge_points(thr)
            # self.get_white_pixels(pixel_matrix)

            return True
        except Exception as e:
            print(e)
            return False    

    
    def find_contour(self)-> bool:
        pixel_matrix = self.get_edge_points()
        if(pixel_matrix is None):
            return False
        if(not(self.get_white_pixels(pixel_matrix))):
            return False
        if(self.contour_to_image() is False):
            return False
       
        if(self.find_edge_corners() is False):
            return False
        # if(self.find_corner() is False):
        #     return False
        # if(self.order_points_clockwise() is False):
        #     return False
        # if(self.fine_tune_corners() is False):
        #     return False
        if(self.find_shape_in_out() is False):
            return False
        if(self.shape_classification() is False):
            return False
        return True


    def get_edge_points(self):
        try:
            threshold_name = join(self.threshold_folder, self.name+".png")
            img1 = cv2.imread(threshold_name, cv2.IMREAD_GRAYSCALE)
            # corners1 = moravec_corner_detection(img1)
            corners1 = harris_corner_detection(img=img1,k=0.06,window_size=7,threshold=0.08)
            cv2.imshow('Corners', corners1)
            cv2.waitKey(0)

            
            img = cv2.imread(threshold_name)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # cv2.imshow("image",image) 
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            pixel_matrix = image.tolist()
            # pixel_matrix = threshold_image(pixel_matrix)
            # Create a copy of the input image
            
            for row in range(1,len(pixel_matrix)-1):
                for col in range(1,len(pixel_matrix[row])-1):
                    if(pixel_matrix[row][col] == 255):
                        if(pixel_matrix[row-1][col] == 0 or \
                           pixel_matrix[row+1][col] == 0 or \
                           pixel_matrix[row][col-1] == 0 or \
                           pixel_matrix[row][col+1] == 0 ):
                            pixel_matrix[row][col] = 128

            for row in range(1,len(pixel_matrix)-1):
                for col in range(1,len(pixel_matrix[row])-1):
                    if(pixel_matrix[row][col] == 128):
                        pixel_matrix[row][col] = 255
                    else:
                        pixel_matrix[row][col] = 0
                                       
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
            return None


    def get_white_pixels(self,pixel_matrix):
        try:     
            pixels_list = []
            for row in range(len(pixel_matrix)):
                for col in range(len(pixel_matrix[row])):
                    if(pixel_matrix[row][col] == 255):
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
                        print(f"{self.name} get_white_pixels Error: {i} {links} {pixels_list[i]}")  
                        return False
                    
                else:
                    print(f"{self.name} get_white_pixels Error: {i} {links} {pixels_list[i]}")
                    return False
            
            # Return the list of white pixels
            for i in range(len(pixels_list)):
                if(status[i] == 0):
                    print(f"{self.name} get_white_pixels Error in  index {i}")
                    return False
            
            start_point = 0
            next_point = left_index[0]
            self.points_list.append(pixels_list[start_point])
           
            while(next_point != 0):
                self.points_list.append(pixels_list[next_point])
                new_point = next_pixel(next_point,start_point,left_index,right_index)
                start_point = next_point
                next_point = new_point
    
            # file_name = join(self.contour_folder, self.name+"_points.csv")
            # f = open(file_name,"w")
      
            # for pt in self.points_list:
            #     f.write(f"{pt[0]},{pt[1]}\n")
                
            # f.close()

            return True
        except Exception as e:
            print(e)
            return False


    
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
        
            file_name = join(self.contour_folder, self.name+".csv")
            f = open(file_name,"w")
            for p in self.points_list:
                f.write(f"{p[0]},{p[1]}\n")
            f.close()

            # list of distacens between points and minx and miny
            dist_list1 = [[distance(pt,[margin,margin]),pt] for pt in self.points_list]
            # list of distacens between points and maxx and miny
            dist_list2 = [[distance(pt,[sizex,margin]),pt] for pt in self.points_list]    
            # list of distances between points and maxx and maxy
            dist_list3 = [[distance(pt,[sizex,sizey]),pt] for pt in self.points_list]
            # list of distances between points and minx and maxy
            dist_list4 = [[distance(pt,[margin,sizey]),pt] for pt in self.points_list]

            # get the index of minimum dist_list1
            index1 = min(enumerate(dist_list1), key=lambda x: x[1])[0]
            # get the index of minimum dist_list2
            index2 = min(enumerate(dist_list2), key=lambda x: x[1])[0]
            # get the index of minimum dist_list3
            index3 = min(enumerate(dist_list3), key=lambda x: x[1])[0]
            # get the index of minimum dist_list4
            index4 = min(enumerate(dist_list4), key=lambda x: x[1])[0]

            self.corners_index = [index1,index2,index3,index4]
            
            
            revers_order = True
            for i in range(4):
                if((self.corners_index[i%4] < self.corners_index[(i+1)%4]) and \
                      (self.corners_index[(i+1)%4] < self.corners_index[(i+2)%4])):
                    revers_order = False

            if revers_order:
                # rotate the self.point_list by self.corners_index[3]
                self.points_list = self.points_list[self.corners_index[3]:] + \
                                    self.points_list[:self.corners_index[3]]
                # update self.corners_index
                self.corners_index = [self.corners_index[0]-self.corners_index[3],\
                                        self.corners_index[1]-self.corners_index[3],\
                                        self.corners_index[2]-self.corners_index[3],\
                                        self.corners_index[3]-self.corners_index[3]]
                list_len = len(self.points_list)
                # reverse self.points_list
                self.points_list = self.points_list[::-1]
                self.corners_index = [0,\
                                        list_len - self.corners_index[0],\
                                        list_len - self.corners_index[1],\
                                        list_len - self.corners_index[2]]
                
            else:
                # rotate the self.point_list by self.corners_index[0]
                self.points_list = self.points_list[self.corners_index[0]:] + \
                                    self.points_list[:self.corners_index[0]]
                self.corners_index = [self.corners_index[0]-self.corners_index[0],\
                                        self.corners_index[1]-self.corners_index[0],\
                                        self.corners_index[2]-self.corners_index[0],\
                                        self.corners_index[3]-self.corners_index[0]]
            
            self.corners = [self.points_list[self.corners_index[0]],\
                            self.points_list[self.corners_index[1]],\
                            self.points_list[self.corners_index[2]],\
                            self.points_list[self.corners_index[3]]]  
            
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


    def find_edge_corners(self)->bool:
        try:
            
            contour_name = join(self.contour_folder, self.name+".png")
            img = cv2.imread(contour_name)
            
            for pt in self.corners:
                color2 = (0,0,255)
                cv2.circle(img,pt,3,color=color2,thickness=1)
                cv2.imshow(self.name,img)

            full_key = cv2.waitKeyEx(0)
            cv2.destroyWindow(self.name)
            if full_key == 27:
                return False
            return True
            
        except Exception as e:
            print(e)
            return False
                                    

    def find_corner(self)->bool:
        try:
            contour_name = join(self.contour_folder, self.name+".png")
            img = cv2.imread(contour_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            corners = cv2.goodFeaturesToTrack(gray, 100, 0.15, 50)
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
            oriatation = self.in_out[0].value - 1
            
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
            
            """
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
 """
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
            
            # blank_image1 = rotate_image(blank_image, angle)
                
            # add "in" or "out" to the file name based on orientation
            if self.in_out[idx] == SideShape.IN:
                filename = f"{self.name}_{idx+1}_in"
            else:
                filename = f"{self.name}_{idx+1}_out"
            cv2.imwrite(join('sides', filename +".png"),blank_image)
            # cv2.imwrite(join('sides', filename +".jpg"),blank_image1)
            
            
            f = open(join('sides', filename +".csv"),"w")
            for p in points:
                f.write(f"{p[0]},{p[1]}\n")
            f.close()

            return True
        except Exception as e:
            print(e)
            return False     

    
    

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
