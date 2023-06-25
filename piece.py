# piece dataclass

import os
from os.path import join
import numpy as np
import pandas as pd
import cv2
import math
from dataclasses import dataclass, field
# from utils import get_line_through_points, distance_point_line_squared,\
#       distance,rotate_points,point_exists,slope_in_degrees
import utils
# from side_extractor import rotation_matrix
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
    Corner = 3

rotation_matrix = np.array([[-1, 0, 1, 2], [1, 2, -1, 0]])

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
    side_corners: list[list] = field(init=False, repr=False,default_factory=list)
    status: str = field(init=False, repr=False, default= "n")
    # edge_points: list[list] = field(init=False, repr=False, default_factory=list)
    # corners: list[list] = field(init=False, repr=False, default_factory=list)
    points_list: list[list] = field(init=False, repr=False, default_factory=list)
    in_out: list[SideShape] = field(init=False, repr=False, default_factory=list)
    # size of the piece
    width: int = field(init=False, repr=False, default= 0)
    height: int = field(init=False, repr=False, default= 0)
    piece_geometry: dict = field(init=False, repr=False, default_factory=dict)
    
    def __post_init__(self):
        self.piece_file_name = self.name + ".jpg"
        os.makedirs(self.piece_folder, exist_ok=True)
        os.makedirs(self.threshold_folder, exist_ok=True)
        os.makedirs(self.contour_folder, exist_ok=True)
        self.corners = []
        self.sides = []


    def camera_image_to_threshold(self,input_filename: str,\
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

            gray1 = cv2.filter2D(gray,-1,kernel_9x9)
            ret, thr = cv2.threshold(gray1, 120, 255, cv2.THRESH_BINARY)  
            
            cv2.imwrite(piece_name, img)
            cv2.imwrite(threshold_name, thr)

            
            return True
        except Exception as e:
            print(e)
            return False    

    
        
    def threshold_to_transparent(self):
        try:
            threshold_name = join(self.threshold_folder, self.name+".png")
            transparent_name = join("transparent", self.name+".png")
            img = cv2.imread(threshold_name)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite(transparent_name,image)
        except Exception as e:
            print(e)
            return False 
    
    
    def threshold_to_contours(self,width: int, height: int)-> bool:
        pixel_matrix = self.get_edge_points()
        if(pixel_matrix is None):
            return False
        if(self.get_white_pixels(pixel_matrix) is False):
            return False
        if(self.contour_to_image(width=width, height=height) is False):
            return False
        
        return True

    def contour_to_corner(self,width: int, height: int)-> bool:
        pixel_matrix = self.get_edge_points()
        if(pixel_matrix is None):
            return False
        if(self.get_white_pixels(pixel_matrix) is False):
            return False
        if(self.contour_to_image(width=width, height=height) is False):
            return False
        
        if(self.show_edge_corners() is False):
            return False
        # # if(self.find_corner() is False):
        # #     return False
        if(self.order_points_clockwise() is False):
            return False
        # # if(self.fine_tune_corners() is False):
        # #     return False
        # if(self.find_shape_in_out() is False):
        #     return False
        # if(self.shape_classification() is False):
        #     return False
        return True


    def get_edge_points(self):
        try:
            threshold_name = join(self.threshold_folder, self.name+".png")
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
                links = utils.get_adjacent_points(pixels_list,i)
                if(len(links) == 2):
                    status[i] = 1
                    left_index[i] = links[0]
                    right_index[i] = links[1]
                elif(len(links) == 3):
                    p0 = pixels_list[i]
                    p1 = pixels_list[links[0]]
                    p2 = pixels_list[links[1]]
                    p3 = pixels_list[links[2]]
                    if(utils.pixel_distance(p0,p1) == 1) and (utils.pixel_distance(p0,p2) == 1):
                        status[i] = 1
                        left_index[i] = links[0]
                        right_index[i] = links[1]
                    elif(utils.pixel_distance(p0,p2) == 1) and (utils.pixel_distance(p0,p3) == 1):
                        status[i] = 1
                        left_index[i] = links[1]
                        right_index[i] = links[2]
                    elif(utils.pixel_distance(p0,p1) == 1) and (utils.pixel_distance(p0,p3) == 1):
                        status[i] = 1
                        left_index[i] = links[0]
                        right_index[i] = links[2]
                    elif(utils.pixel_distance(p0,p1) == 2) and (utils.pixel_distance(p0,p2) == 2):
                        status[i] = 1
                        left_index[i] = links[2]
                        if(utils.pixel_distance(p1,p3) < utils.pixel_distance(p2,p3)):
                            right_index[i] = links[1]
                        else:
                            right_index[i] = links[0]
                    elif(utils.pixel_distance(p0,p1) == 2) and (utils.pixel_distance(p0,p3) == 2):
                        status[i] = 1
                        left_index[i] = links[1]
                        if(utils.pixel_distance(p1,p2) < utils.pixel_distance(p3,p2)):
                            right_index[i] = links[2]
                        else:
                            right_index[i] = links[0]
                    elif(utils.pixel_distance(p0,p2) == 2) and (utils.pixel_distance(p0,p3) == 2):
                        status[i] = 1
                        left_index[i] = links[0]
                        if(utils.pixel_distance(p2,p1) < utils.pixel_distance(p3,p1)):
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
                new_point = utils.next_pixel(next_point,start_point,left_index,right_index)
                start_point = next_point
                next_point = new_point
    
           
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
    def contour_to_image(self,width: int, height: int)-> bool:
        try:
            # blank_image = np.zeros((max_size(0),max_size(1),3), np.uint8)
            # minimum value of x and y
            minx = min(self.points_list, key=lambda x: x[0])[0]
            miny = min(self.points_list, key=lambda x: x[1])[1]
            # maximum value of x and y
            maxx = max(self.points_list, key=lambda x: x[0])[0]
            maxy = max(self.points_list, key=lambda x: x[1])[1]
            # margin = 5
            margin_x = (width-(maxx - minx))//2 
            margin_y = (height-(maxy - miny))//2
            # size of the image
            # sizex, sizey = ((maxx - minx + margin*2), (maxy - miny+margin*2))

            blank_image = np.zeros((width, height, 3), np.uint8)
            self.width, self.height = blank_image.shape[:2]
            # self.edge_points = []
            # draw the contour

            for i in range(len(self.points_list) ):
                self.points_list[i] = [self.points_list[i][0] - minx + margin_x, \
                        self.points_list[i][1] - miny + margin_y]
        
            

            # self.get_corners_from_pointlist(width, height)
            # self.fine_tune_corners()
            
            file_name = join(self.contour_folder, self.name+".csv")
            f = open(file_name,"w")
            for p in self.points_list:
                f.write(f"{p[0]},{p[1]}\n")
            f.close()

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

            # self.get_corners_from_pointlist(sizex=width, sizey=height)
            # self.fine_tune_corners()
            
            return True
        
        except Exception as e:
            print(str(e))
            return False

    def corner_detect(self,width: int, height: int)-> bool:
        return self.get_corners_from_pointlist(width, height)
        
        
    def show_side_points(self)->bool:
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
        
    def load_points_list(self):
        try:
            file_name = join(self.contour_folder, self.name+".csv")
            f = open(file_name,"r")
            self.points_list = []
            for line in f:
                self.points_list.append([int(x) for x in line.split(",")])
            f.close()
            return True
        except Exception as e:
            print(str(e))
            return False      

    def load_second_points_list(self):
        try:
            file_name = join(self.contour_folder, self.name+"_angle.csv")
            f = open(file_name,"r")
            self.points_list = []
            for line in f:
                sep = line.split(",")
                self.points_list.append((int(sep[0]),int(sep[1])))
            f.close()
            return True
        except Exception as e:
            print(str(e))
            return False      

    def get_corners_from_pointlist(self, sizex, sizey)->bool:
        try:
            self.load_points_list()
            # list of distacens between points and minx and miny
            dist_list1 = [[utils.distance(pt,[0,0]),pt] for pt in self.points_list]
            # list of distacens between points and maxx and miny
            dist_list2 = [[utils.distance(pt,[sizex,0]),pt] for pt in self.points_list]    
            # list of distances between points and maxx and maxy
            dist_list3 = [[utils.distance(pt,[sizex,sizey]),pt] for pt in self.points_list]
            # list of distances between points and minx and maxy
            dist_list4 = [[utils.distance(pt,[0,sizey]),pt] for pt in self.points_list]

            # get the index of minimum dist_list1
            index1 = min(enumerate(dist_list1), key=lambda x: x[1])[0]
            # get the index of minimum dist_list2
            index2 = min(enumerate(dist_list2), key=lambda x: x[1])[0]
            # get the index of minimum dist_list3
            index3 = min(enumerate(dist_list3), key=lambda x: x[1])[0]
            # get the index of minimum dist_list4
            index4 = min(enumerate(dist_list4), key=lambda x: x[1])[0]

            self.corners_index = [index1,index2,index3,index4]
            self.arrange_points()
            # list of distacens between points and minx and miny
            dist_list1 = [[utils.distance(pt,[0,0]),pt] for pt in self.points_list]
            # list of distacens between points and maxx and miny
            dist_list2 = [[utils.distance(pt,[sizex,0]),pt] for pt in self.points_list]    
            # list of distances between points and maxx and maxy
            dist_list3 = [[utils.distance(pt,[sizex,sizey]),pt] for pt in self.points_list]
            # list of distances between points and minx and maxy
            dist_list4 = [[utils.distance(pt,[0,sizey]),pt] for pt in self.points_list]

            # get the index of minimum dist_list1
            index1 = min(enumerate(dist_list1), key=lambda x: x[1])[0]
            # get the index of minimum dist_list2
            index2 = min(enumerate(dist_list2), key=lambda x: x[1])[0]
            # get the index of minimum dist_list3
            index3 = min(enumerate(dist_list3), key=lambda x: x[1])[0]
            # get the index of minimum dist_list4
            index4 = min(enumerate(dist_list4), key=lambda x: x[1])[0]


            angle_threshold = 45
            # distance_threshold = 20
            file_name = join(self.contour_folder, self.name+"_angle.csv")
            f = open(file_name,"w")
            
            # df = pd.DataFrame()
            # size of points_list
            list_len = len(self.points_list)
            temp_list = []
            for i, p in enumerate(self.points_list):
                angle = utils.find_lines_interpolate_and_angle(self.points_list,i)
                
                # new_row = pd.DataFrame({'Index': i,'X':p[0],\
                #                         'Y':p[1],'angle':angle, },index=[0])
                # df = pd.concat([df, new_row], axis=0, ignore_index=True)    
                # df = df.append(new_row)
                f.write(f"{p[0]},{p[1]},{angle}\n")
                if(angle > angle_threshold):
                    temp_list.append(p)
                else:
                    temp_list.append([0,0])
            f.close()
                
            
            # list of distacens between points and minx and miny
            dist_list1 = [utils.distance(pt,[0,0]) for pt in temp_list]
            # list of distacens between points and maxx and miny
            dist_list2 = [utils.distance(pt,[sizex,0]) for pt in temp_list]    
            # list of distances between points and maxx and maxy
            dist_list3 = [utils.distance(pt,[sizex,sizey]) for pt in temp_list]
            # list of distances between points and minx and maxy
            dist_list4 = [utils.distance(pt,[0,sizey]) for pt in temp_list]

            # get the index of minimum dist_list1 for non zero values
            non_zero_list = list(filter(lambda x: x != 0, dist_list1))
            # Use min to find the minimum value of the non-zero elements
            min_non_zero = min(non_zero_list)
            # Use index to get the index of the minimum value of the non-zero elements
            index1 = dist_list1.index(min_non_zero)
            # check if index1 is a list get the first element otherwise return index1
            index1 = index1[0] if isinstance(index1, list) else index1

            # get the index of minimum dist_list1 for non zero values
            non_zero_list = list(filter(lambda x: x != 0, dist_list2))
            # Use min to find the minimum value of the non-zero elements
            min_non_zero = min(non_zero_list)
            # Use index to get the index of the minimum value of the non-zero elements
            index2 = dist_list2.index(min_non_zero)
            # check if index1 is a list get the first element otherwise return index2
            index2 = index2[0] if isinstance(index2, list) else index2
        
            # get the index of minimum dist_list1 for non zero values
            non_zero_list = list(filter(lambda x: x != 0, dist_list3))
            # Use min to find the minimum value of the non-zero elements
            min_non_zero = min(non_zero_list)
            # Use index to get the index of the minimum value of the non-zero elements
            index3 = dist_list3.index(min_non_zero)
            # check if index1 is a list get the first element otherwise return index3
            index3 = index3[0] if isinstance(index3, list) else index3

            # get the index of minimum dist_list1 for non zero values
            non_zero_list = list(filter(lambda x: x != 0, dist_list4))
            # Use min to find the minimum value of the non-zero elements
            min_non_zero = min(non_zero_list)
            # Use index to get the index of the minimum value of the non-zero elements
            index4 = dist_list4.index(min_non_zero)
            # check if index1 is a list get the first element otherwise return index4
            index4 = index4[0] if isinstance(index4, list) else index4


            

            self.corners_index = [index1,index2,index3,index4]
            self.corners = [self.points_list[self.corners_index[0]],\
                            self.points_list[self.corners_index[1]],\
                            self.points_list[self.corners_index[2]],\
                            self.points_list[self.corners_index[3]]]  

            blank_image = np.zeros((sizex, sizey, 3), np.uint8)
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
                cv2.line(blank_image, pt1, pt2, (255, 255, 255), 2)
            
            i = 0
            sides = []

            sides.append(self.corners_index[0])
            sides.append(self.corners_index[1])
            sides.append(self.corners_index[1])
            sides.append(self.corners_index[2])
            sides.append(self.corners_index[2])
            sides.append(self.corners_index[3])
            sides.append(self.corners_index[3])
            sides.append(self.corners_index[0])

            side_index = 0
            full_key = 0
            color1 = (255,255,255)
            color2 = (0,0,255)
            # loop until operator press ESC key
            # if operator press Space go to the next sides index 
            # and by pressing 1 increase the index and by pressing 2 decrease the index
            # and show the point on the image
            index1 = sides[side_index]
            while(full_key != 13):
                
                cv2.circle(blank_image,self.points_list[index1],1,color=color2,thickness=1)    
                cv2.imshow(self.name,blank_image)
                full_key = cv2.waitKeyEx(0)
                cv2.circle(blank_image,self.points_list[index1],1,color=color1,thickness=1)    
                cv2.imshow(self.name,blank_image)
                # print(f"Key pressed {full_key}\n")
                
                if full_key == 27:
                    cv2.destroyWindow(self.name)
                    return False
                if full_key == 32:
                    side_index = (side_index +1)  % len(sides)
                    index1 = sides[side_index]
                if full_key == 49:
                    index1 += 1
                    index1 = index1 % len(self.points_list)
                    sides[side_index] = index1
                if full_key == 50:
                    index1 -= 1
                    index1 = (index1 + len(self.points_list)) % len(self.points_list)
                    sides[side_index] = index1

            index1 = sides[0]
            index2 = sides[1]
            self.side_corners.append(((index1,self.points_list[index1][0],self.points_list[index1][1]),\
                                      (index2,self.points_list[index2][0],self.points_list[index2][1])))   
            index1 = sides[2]
            index2 = sides[3]
            self.side_corners.append(((index1,self.points_list[index1][0],self.points_list[index1][1]),\
                                        (index2,self.points_list[index2][0],self.points_list[index2][1])))
            index1 = sides[4]
            index2 = sides[5]
            self.side_corners.append(((index1,self.points_list[index1][0],self.points_list[index1][1]),\
                                        (index2,self.points_list[index2][0],self.points_list[index2][1])))
            index1 = sides[6]
            index2 = sides[7]
            self.side_corners.append(((index1,self.points_list[index1][0],self.points_list[index1][1]),\
                                        (index2,self.points_list[index2][0],self.points_list[index2][1])))
           



            cv2.destroyAllWindows()
            return True
        except Exception as e:
            print(f"Error in show_edge_corners {e}")
            return False
        

    def show_corners(self,X1,Y1,X2,Y2,X3,Y3,X4,Y4):
        self.load_points_list()
        self.corners = [[X1,Y1],[X2,Y2],[X3,Y3],[X4,Y4]]
        self.fine_tune_corners()
        self.show_edge_corners()

    def show_edge_corners(self)->bool:
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

            

            # revers_order = True
            # for i in range(4):
            #     if((self.corners_index[i%4] < self.corners_index[(i+1)%4]) and \
            #           (self.corners_index[(i+1)%4] < self.corners_index[(i+2)%4])):
            #         revers_order = False
            
            # temp_list_corners = []
            
            # temp_list_piece = []
            # no_of_points = len(self.points_list)
            # for i in range(4):
            #     start_index = self.corners_index[i]
            #     temp_list_corners.append(len(temp_list_piece))
            #     temp_list_side = []
                
            #     last_index = self.corners_index[(i+1)%4]

            #     while(start_index != last_index):
            #         temp_list_side.append(self.points_list[start_index])
            #         if revers_order:
            #             start_index = (start_index+no_of_points-1)%no_of_points
            #         else:
            #             start_index = (start_index+1)%no_of_points

            #     temp_list_piece = temp_list_piece + temp_list_side

                

            # self.corners_index = temp_list_corners
            # self.points_list = temp_list_piece
            
            return True
        except Exception as e:
            print(e)
            return False
        
    
    def find_nearest_point(self, idx, a,b)->bool:
        try:
            """Finds the nearest point in a contour to a given point."""
            point = self.corners[idx]

            # find the point in point_list that is closest to point
            # closest_index = utils.find_closest_point_index(self.points_list, point)
            # self.corners_index.insert(idx, closest_index)
            # x,y = self.points_list[closest_index]
            x = point[0]
            y = point[1]
            
            while utils.point_exists(self.points_list, x+a,y+b):
                x = x+a
                y = y+b
            while utils.point_exists(self.points_list, x+a,y):
                x = x+a
            while utils.point_exists(self.points_list, x,y+b):
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

    # reange the points in the list to be in the right order
    def arrange_points(self)->bool:
        
        revers_order = True
        for i in range(4):
            if((self.corners_index[i%4] < self.corners_index[(i+1)%4]) and \
                    (self.corners_index[(i+1)%4] < self.corners_index[(i+2)%4])):
                revers_order = False

        if revers_order:
            # rotate the self.point_list by self.corners_index[3]
            # self.points_list = self.points_list[self.corners_index[3]+1:] + \
            #                     self.points_list[:self.corners_index[3]+1]
            
            
            # reverse self.points_list
            self.points_list = self.points_list[::-1]
            
            
            
        # else:
        #     # rotate the self.point_list by self.corners_index[0]
        #     self.points_list = self.points_list[self.corners_index[0]:] + \
        #                         self.points_list[:self.corners_index[0]]
            
        
        # find index of point in self.points_list whic is equal X1 and Y1
        """ X1 = self.corners[0][0]
        Y1 = self.corners[0][1]
        X2 = self.corners[1][0]
        Y2 = self.corners[1][1]
        X3 = self.corners[2][0]
        Y3 = self.corners[2][1]
        X4 = self.corners[3][0]
        Y4 = self.corners[3][1]
        
        self.corners_index[0] = [i for i,point in enumerate(self.points_list) if point == [X1,Y1]][0]
        self.points_list = self.points_list[self.corners_index[0]:] + \
                                self.points_list[:self.corners_index[0]]
        self.corners_index[0] = [i for i,point in enumerate(self.points_list) if point == [X1,Y1]][0]
        # find index of point in self.points_list whic is equal X2 and Y2
        self.corners_index[1] = [i for i,point in enumerate(self.points_list) if point == [X2,Y2]][0]
        # find index of point in self.points_list whic is equal X3 and Y3
        self.corners_index[2] = [i for i,point in enumerate(self.points_list) if point == [X3,Y3]][0]
        # find index of point in self.points_list whic is equal X4 and Y4
        self.corners_index[3] = [i for i,point in enumerate(self.points_list) if point == [X4,Y4]][0]

        self.corners = [self.points_list[self.corners_index[0]],\
                        self.points_list[self.corners_index[1]],\
                        self.points_list[self.corners_index[2]],\
                        self.points_list[self.corners_index[3]]]   """


    def clasification(self,X1,Y1,X2,Y2,X3,Y3,X4,Y4) -> bool:
        if(self.find_shape_in_out(X1,Y1,X2,Y2,X3,Y3,X4,Y4) is False):
            return False
        if(self.shape_classification() is False):
            return False
        return True

    def side(self, sizex, sizey)->bool:
        if(self.load_second_points_list() is False):
            return False
        if(self.find_shape_in_out(sizex, sizey) is False):
            return False
       
        return True

    def find_shape_in_out(self, sizex: int, sizey: int)->bool:
        try:
            # convert to 3.4 to int

            
            index1_1 = int(self.side_corners[0][0][0])
            index1_2 = int(self.side_corners[0][1][0])
            index2_1 = int(self.side_corners[1][0][0])
            index2_2 = int(self.side_corners[1][1][0])
            index3_1 = int(self.side_corners[2][0][0])
            index3_2 = int(self.side_corners[2][1][0])
            index4_1 = int(self.side_corners[3][0][0])
            index4_2 = int(self.side_corners[3][1][0])


            self.corners = []
            self.corners.append((self.side_corners[0][0][1],self.side_corners[0][0][2]))
            self.corners.append((self.side_corners[1][0][1],self.side_corners[1][0][2]))
            self.corners.append((self.side_corners[2][0][1],self.side_corners[2][0][2]))
            self.corners.append((self.side_corners[3][0][1],self.side_corners[3][0][2]))
            
            # get the center of the xy
            center = [sum(pt[0] for pt in self.corners) // len(self.corners),\
                       sum(pt[1] for pt in self.corners) // len(self.corners)]

            side_points = []
            buffer_len = len(self.points_list)
            t = []
            if(index1_1 < index1_2):
                side_points.append(self.points_list[index1_1:index1_2])
            else:
                t = self.points_list[index1_1:buffer_len]
                t.extend(self.points_list[0:index1_2])
                side_points.append(t)
                # side_points.append(self.points_list[index1_1:buffer_len])
                # side_points.append(self.points_list[0:index1_2])
            if(index2_1 < index2_2):
                side_points.append(self.points_list[index2_1:index2_2])
            else:
                t = self.points_list[index2_1:buffer_len]
                t.extend(self.points_list[0:index2_2])
                side_points.append(t)
                # side_points.append(self.points_list[index2_1:buffer_len])
                # side_points.append(self.points_list[0:index2_2])
            if(index3_1 < index3_2):
                side_points.append(self.points_list[index3_1:index3_2])
            else:
                t = self.points_list[index3_1:buffer_len]
                t.extend(self.points_list[0:index3_2])
                side_points.append(t)
                # side_points.append(self.points_list[index3_1:buffer_len])
                # side_points.append(self.points_list[0:index3_2])
            if(index4_1 < index4_2):
                side_points.append(self.points_list[index4_1:index4_2])
            else:
                t = self.points_list[index4_1:buffer_len]
                t.extend(self.points_list[0:index4_2])
                side_points.append(t)
                # side_points.append(self.points_list[index4_1:buffer_len])
                # side_points.append(self.points_list[0:index4_2])

            for i in range(4):
                
                line = utils.get_line_through_points(self.corners[i],self.corners[(i+1)%4])
                # find the index of maximum distance between points in side_points and line
                ds = [utils.distance_point_line_squared(line, point) for point in side_points[i]]
                max_d = max(ds)
                max_d_index = ds.index(max_d)
                corner1 = self.corners[i]
                corner2 = self.corners[(i+1)%4]    
                pt1 = [(corner1[0]+corner2[0])/2,(corner1[1]+corner2[1])/2]
                d1 = utils.distance(pt1,center)
                d2 = utils.distance(side_points[i][max_d_index],center)
                if(d1<d2):
                    self.in_out.append(SideShape.OUT)
                else:
                    self.in_out.append(SideShape.IN)
            

            
            for i in range(4):
                if self.in_out[i] == SideShape.IN:
                    filename = f"{self.name}_{i+1}_in"
                else:
                    filename = f"{self.name}_{i+1}_out"
                self.piece_geometry[filename] = get_geometry(self.name,side_points[i])


            for i in range(4):
                self.side_to_image(i,side_points[i], sizex, sizey)
            return True
        except Exception as e:
            print(e)
            return False   

    # def shape_classification(self)->bool:
        
    #     try:
    #         for i in range(4):
    #             if(i == 3):
    #                 side_points = self.points_list[self.corners_index[i]:]
    #             else:
    #                 side_points = self.points_list[self.corners_index[i]:self.corners_index[(i+1)%4]]
    #             self.side_to_image(i,side_points)
    #         return True
    #     except Exception as e:
    #         print(e)
    #         return False

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


    def side_to_image(self, idx: int,points: list,sizex: int, sizey: int)->bool:
        try:
            # show_point(points)
            oriatation = self.in_out[0].value - 1
            
            # points = utils.rotate_points(points, \
            #                     rotation_matrix[oriatation, idx], \
            #                         self.corners[idx])
            
           
            
            blank_image = np.zeros((sizey, sizex, 3), np.uint8)
            
            for i in range(len(points) -1 ):
                
                pt1 = points[i]
                pt2 = points[i+1]
                cv2.line(blank_image, pt1, pt2, (255, 255, 255), 1)
            
            # add "in" or "out" to the file name based on orientation
            if self.in_out[idx] == SideShape.IN:
                filename = f"{self.name}_{idx+1}_in"
            else:
                filename = f"{self.name}_{idx+1}_out"
            rot = rotation_matrix[oriatation, idx]
            if(rot >= 0):
                blank_image1 = cv2.rotate(blank_image, rot)
                cv2.imwrite(join('sides', filename +".jpg"),blank_image1)
            else:
                cv2.imwrite(join('sides', filename +".jpg"),blank_image)
            

            return True
        except Exception as e:
            print(e)
            return False     

    
    

def get_geometry(name: str, points:list):
    try:
        geometry = {}
        geometry["piece_name"] = name
        # remove the first 5 points and the last 5 points
        # points = points[5:-5]

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
        dist = [utils.distance_point_line_squared(a_b_c=line, x0_y0=point) \
                for point in points] 

        # find the max distance and its index
        max_dist = max(dist)
        
        # return list of indexes of max_dist
        max_index = [i for i, j in enumerate(dist) if j == max_dist]

        max_len = len(max_index)

        if(len(max_index) > 1):
            print("more than one max_dist")
        # find the middle point of the max_dist points
        max_point = points[max_index[max_len//2]]

        geometry["Height"] = max_dist
        # find the intercept point of ortagonal line from max_point and line
        x1,y1 = utils.find_intersection(max_point[0], max_point[1],\
            line[0], line[1], line[2])

        symetry = utils.distance(points[0], (x1,y1)) / width
        geometry['symetry'] = symetry

        

        thr = 20
        idx = []
        idx.append(get_point1(points,line,max_index[max_len//2],thr))
        idx.append(get_point2(points,line,max_index[max_len//2],thr))
        idx.append(get_point3(points,line,max_index[max_len//2],thr))
        idx.append(get_point4(points,line,max_index[max_len//2],thr))
        
        for i in range(4):
            if(idx[i] is not None):
                x1,y1 = utils.find_intersection(points[idx[i]][0], points[idx[i]][1],\
                    line[0], line[1], line[2])
                d = utils.distance(points[0], (x1,y1)) / width
                geometry['d'+str(i+1)] = d
                h = utils.distance_point_line_squared(a_b_c=line, x0_y0=(x1,y1))
                geometry['h'+str(i+1)] = h

        # return the geometry
        return geometry
    except Exception as e:
        print(str(e))
        return None

def get_point1(points:list,line:list,max_index:int,thr:int):
    for i in range (len(points)):
        if(i < max_index):
            l1,l2,l3 = utils.find_orthagonal(x=points[i][0], y=points[i][1],\
                                                a=line[0], b=line[1], c=line[2])
            overlap = [l1*point[0] + l2*point[1] + l3 for point in points]
            
            min_val = 100000
            min_index = 0
            for j in range(len(points)):
                if(j < max_index):
                    # check if absoulte value of overlap is les than 10 the add to the cross list
                    if(abs(overlap[j]) < thr) and (abs(i - j) > 5):
                        if( min_val > abs(overlap[j])) :
                            min_val = abs(overlap[j])
                            min_index = j
                            break
                        
            if(min_val != 100000):
                # print(f" points[{min_index}] = {points[min_index]}")
                return min_index
    return None

def get_point2(points,line,max_index,thr):
    for i in range (len(points)-1,-1,-1):
        if(i > max_index):
            l1,l2,l3 = utils.find_orthagonal(x=points[i][0], y=points[i][1],\
                                                a=line[0], b=line[1], c=line[2])
            overlap = [l1*point[0] + l2*point[1] + l3 for point in points]
            
            min_val = 100000
            min_index = 0
            for j in range(len(points)-1,-1,-1):
                if(j > max_index):
                    # check if absoulte value of overlap is les than 10 the add to the cross list
                    if(abs(overlap[j]) < thr)  and (abs(i-j) > 5):
                        if( min_val > abs(overlap[j])) :
                            min_val = abs(overlap[j])
                            min_index = j
                            break
                        
            if(min_val != 100000):
                # print(f" points[{min_index}] = {points[min_index]}")
                return min_index
    return None

def get_point3(points,line,max_index,thr):
    for i in range (len(points)-1,-1,-1):
        if(i < max_index):
            l1,l2,l3 = utils.find_orthagonal(x=points[i][0], y=points[i][1],\
                                                a=line[0], b=line[1], c=line[2])
            overlap = [l1*point[0] + l2*point[1] + l3 for point in points]
            
            min_val = 100000
            min_index = 0
            for j in range(len(points)-1,-1,-1):
                if(j < max_index):
                    # check if absoulte value of overlap is les than 10 the add to the cross list
                    if(abs(overlap[j]) < thr) and  (abs(i - j) > 5):
                        if( min_val > abs(overlap[j])) :
                            min_val = abs(overlap[j])
                            min_index = j
                            break
                        
            if(min_val != 100000):
                # print(f" points[{min_index}] = {points[min_index]}")
                return min_index
    return None

def get_point4(points,line,max_index,thr):
    for i in range (len(points)):
        if(i > max_index):
            l1,l2,l3 = utils.find_orthagonal(x=points[i][0], y=points[i][1],\
                                                a=line[0], b=line[1], c=line[2])
            overlap = [l1*point[0] + l2*point[1] + l3 for point in points]
            
            min_val = 100000
            min_index = 0
            for j in range(len(points)):
                if(j > max_index):
                    # check if absoulte value of overlap is les than 10 the add to the cross list
                    if(abs(overlap[j]) < thr) and (abs(i - j) > 5):
                        if( min_val > abs(overlap[j])) :
                            min_val = abs(overlap[j])
                            min_index = j
                            break
                        
            if(min_val != 100000):
                # print(f" points[{min_index}] = {points[min_index]}")
                return min_index
    return None