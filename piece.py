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

@dataclass
class Piece:
    piece_file_name: str = field(init=False, repr=False)
    name: str
    corners: list[list] = field(init=False, repr=False,default_factory=list)
    sides: list[Side] = field(init=False, repr=False,default_factory=list)
    camera_folder: str =  field(init=False, repr=False, default= "camera")
    piece_folder: str = field(init=False, repr=False, default= "pieces")
    threshold_folder: str = field(init=False, repr=False, default= "threshold")
    contour_folder: str = field(init=False, repr=False, default= "contours")
    
    status: str = field(init=False, repr=False, default= "n")
    edge_points: list[list] = field(init=False, repr=False, default_factory=list)
    # corners: list[list] = field(init=False, repr=False, default_factory=list)
    points_list: list[list] = field(init=False, repr=False, default_factory=list)
    in_out: list[int] = field(init=False, repr=False, default_factory=list)

    
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
        img = cv2.GaussianBlur(img,(3,3),0)
        ret, thr = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)  
        piece_name = join(self.piece_folder, self.name+".png")
        cv2.imwrite(piece_name, img)
        threshold_name = join(self.threshold_folder, self.name+".png")
        cv2.imwrite(threshold_name, thr)
        if(self.contour_to_points(image=thr) is False):
            return False    
        gray = self.contour_to_image()
        if(self.find_corner(gray) is False):
            return False
        if(self.order_points_clockwise() is False):
            return False
        if(self.fine_tune_corners() is False):
            return False
           
        if(self.shape_classification(gray.shape) is False):
            return False
        return True

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
    def contour_to_image(self):
        try:
            # blank_image = np.zeros((max_size(0),max_size(1),3), np.uint8)
            # minimum value of x and y
            minx = min(self.points_list, key=lambda x: x[0])[0]
            miny = min(self.points_list, key=lambda x: x[1])[1]
            # maximum value of x and y
            maxx = max(self.points_list, key=lambda x: x[0])[0]
            maxy = max(self.points_list, key=lambda x: x[1])[1]
            margin = 50
            # size of the image
            sizex, sizey = ((maxx - minx + margin*2), (maxy - miny+margin*2))

            blank_image = np.zeros((sizey, sizex, 3), np.uint8)
 
            self.edge_points = []
            # draw the contour
            for i in range(len(self.points_list) ):
                index1 = i % len(self.points_list)
                index2 = (i+1) % len(self.points_list)
                
                pt1 = [self.points_list[index1][0] - minx + margin, \
                        self.points_list[index1][1] - miny + margin]
                pt2 = [self.points_list[index2][0] - minx + margin, \
                        self.points_list[index2][1] - miny + margin]
                self.edge_points.append(pt1)
                cv2.line(blank_image, pt1, pt2, (255, 255, 255), 1)

            ret, gray = cv2.threshold(blank_image, 128, 255, cv2.THRESH_BINARY) 

            # cv2.drawContours(blank_image, new_points, -1, (255,255,255), 1)
            contour_name = join(self.contour_folder, self.name+".png")
            cv2.imwrite(contour_name,gray)
            return gray
        
        except Exception as e:
            print(str(e))

     
    def find_corner(self, img1)->bool:
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
            contour_name = join(self.contour_folder, self.name+".png")
            img = cv2.imread(contour_name)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Threshold to get white pixels
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

            # Find white pixel coordinates
            white_pixels = np.where(thresh == 255)
            coords = [[x, y] for x, y in zip(white_pixels[1], white_pixels[0])]
            
            # find the closest point of coords to x,y
            self.find_nearest_point(coords, 0,-1,-1)
            self.find_nearest_point(coords, 1,1,-1)
            self.find_nearest_point(coords, 2,1,1)
            self.find_nearest_point(coords, 3,-1,1)
            return True
        except Exception as e:
            print(e)
            return False
        
    


    def find_nearest_point(self,points, idx, a,b)->bool:
        try:
            """Finds the nearest point in a contour to a given point."""
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
            return False

       

    def shape_classification(self,img_shape)->bool:
        
        try:
            threshhold = 30

            # get the center of the xy
            # center = np.mean(out_dict['xy'],axis=0)
            # calculate to center point of corners list
            center = [sum(pt[0] for pt in self.corners) // len(self.corners),\
                       sum(pt[1] for pt in self.corners) // len(self.corners)]


            lines = []  
            lines.append(get_line_through_points(self.corners[0],self.corners[1]))
            lines.append(get_line_through_points(self.corners[1],self.corners[2]))
            lines.append(get_line_through_points(self.corners[2],self.corners[3]))
            lines.append(get_line_through_points(self.corners[3],self.corners[0]))


            # class_image = np.zeros(img.shape, dtype='uint8')
            c_points = []

            for _edge in self.edge_points:
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
            blank_image.append(np.zeros(img_shape, np.uint8))
            blank_image.append(np.zeros(img_shape, np.uint8))
            blank_image.append(np.zeros(img_shape, np.uint8))
            blank_image.append(np.zeros(img_shape, np.uint8))


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
            
            for i,points in enumerate(four_sides_points):
                head_point = 0
                head_point_index = -1
                
                for j in range(len(four_sides_points[i])):
                    # find the maximum distance between points in points[i] and lines[i]
                    
                    d = distance_point_line_squared(lines[i], points[j])
                    if (d > head_point):
                        head_point = d
                        head_point_index = j
                corner1 = self.corners[i]
                corner2 = self.corners[(i+1)%4]    
                
                # pt1 is middle point between corner1 and corner2
                pt1 = [(corner1[0]+corner2[0])/2,(corner1[1]+corner2[1])/2]
                d1 = distance(pt1,center)
                d2 = distance(points[head_point_index],center)
                if(d1<d2):
                    self.in_out.append(1)
                else:
                    self.in_out.append(0)
                    
            for i,points in enumerate(four_sides_points):
                self.side_to_image(i,points)
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
            oriatation = self.in_out[0]
            
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
            if self.in_out[idx] == 0:
                filename = f"{self.name}_{idx+1}_in"
            else:
                filename = f"{self.name}_{idx+1}_out"
            cv2.imwrite(join('sides', filename +".png"),blank_image)
            # cv2.imshow("image",blank_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return True
        except Exception as e:
            print(e)
            return False     