# piece dataclass

import os
from os.path import join
import numpy as np
import pandas as pd
import cv2
import math
from PIL import Image
from dataclasses import dataclass, field

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
    X1: int = field(init=False, repr=False, default= -1)
    Y1: int = field(init=False, repr=False, default= -1)
    IO1: int = field(init=False, repr=False, default= -1)
    X2: int = field(init=False, repr=False, default= -1)
    Y2: int = field(init=False, repr=False, default= -1)
    IO2: int = field(init=False, repr=False, default= -1)
    X3: int = field(init=False, repr=False, default= -1)
    Y3: int = field(init=False, repr=False, default= -1)
    IO3: int = field(init=False, repr=False, default= -1)
    X4: int = field(init=False, repr=False, default= -1)
    Y4: int = field(init=False, repr=False, default= -1)
    IO4: int = field(init=False, repr=False, default= -1)
    status: str = field(init=False, repr=False, default= "new")
    new_points: list[list] = field(init=False, repr=False, default_factory=list)
    corners: list[list] = field(init=False, repr=False, default_factory=list)
    points_list: list[list] = field(init=False, repr=False, default_factory=list)


    
    def __post_init__(self):
        self.piece_file_name = self.name + ".jpg"
        
        # self.piece_folder_threshold = join(self.piece_folder_threshold, self.name)
        # self.contour_folder = join(self.contour_folder, self.name)
        os.makedirs(self.piece_folder, exist_ok=True)
        os.makedirs(self.threshold_folder, exist_ok=True)
        os.makedirs(self.contour_folder, exist_ok=True)
        self.corners = []
        self.sides = []

    def read_camera_image(self,input_filename: str):
        img = cv2.imread(join("camera", input_filename))
        img = img[1100:1700,1400:2000]
        img = cv2.GaussianBlur(img,(3,3),0)
        ret, thr = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)  
        piece_name = join(self.piece_folder, self.name+".jpg")
        cv2.imwrite(piece_name, img)
        threshold_name = join(self.threshold_folder, self.name+".jpg")
        cv2.imwrite(threshold_name, thr)
        self.contour_to_points(image=thr)
        gray = self.contour_to_image()
        self.corners = self.find_corner(gray)
        self.corners = self.order_points_clockwise(self.corners)
        self.find_white_pixels()
        # new_row = pd.DataFrame({'piece':self.name, \
        #     'status':'new', \
        #     'X1':-1, \
        #     'Y1':-1, \
        #     'IO1':-1, \
        #     'X2':-1, \
        #     'Y2':-1, \
        #     'IO2':-1, \
        #     'X3':-1, \
        #     'Y3':-1, \
        #     'IO3':-1, \
        #     'X4':-1, \
        #     'Y4':-1, \
        #     'IO4':-1, \
        #         },index=[0])
        # df_pieces = pd.concat([new_row,df_pieces.loc[:]]).reset_index(drop=True)
        # return df_pieces

    def contour_to_points(self,image):
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

        except Exception as e:
            print(e)

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
 
            self.new_points = []
            # draw the contour
            for i in range(len(self.points_list) ):
                index1 = i % len(self.points_list)
                index2 = (i+1) % len(self.points_list)
                
                pt1 = [self.points_list[index1][0] - minx + margin, \
                        self.points_list[index1][1] - miny + margin]
                pt2 = [self.points_list[index2][0] - minx + margin, \
                        self.points_list[index2][1] - miny + margin]
                self.new_points.append(pt1)
                cv2.line(blank_image, pt1, pt2, (255, 255, 255), 1)

            ret, gray = cv2.threshold(blank_image, 128, 255, cv2.THRESH_BINARY) 

            # cv2.drawContours(blank_image, new_points, -1, (255,255,255), 1)
            contour_name = join(self.contour_folder, self.name+".jpg")
            cv2.imwrite(contour_name,gray)
            return gray
        
        except Exception as e:
            print(str(e))

     
    def find_corner(self, img1):

        contour_name = join(self.contour_folder, self.name+".jpg")
        img = cv2.imread(contour_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.2, 10)
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
            l = len(corners)
            output_list = []
            count = 0
            for index, row in df.iterrows():
                x = int(row['X'])
                y = int(row['Y'])
                xy = (x,y)
                
                color2 = (0,0,255)
                cv2.circle(img,xy,1,color=color2,thickness=1)
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
            
            # print(df)
        return output_list
    

    def order_points_clockwise(self,pts):
        # Initialize the list of ordered points
        ordered_pts = [None] * 4
        
        # Find the center of the points
        center = [sum(pt[0] for pt in pts) // len(pts), sum(pt[1] for pt in pts) // len(pts)]
        
        # Divide the points into two groups: those above the center and those below
        above_center = []
        below_center = []
        for pt in pts:
            if pt[1] < center[1]:
                above_center.append(pt)
            else:
                below_center.append(pt)
        
        # Sort the points in each group by their x-coordinate
        above_center = sorted(above_center, key=lambda pt: pt[0])
        below_center = sorted(below_center, key=lambda pt: pt[0], reverse=True)
        
        # Assign the ordered points to the output list
        ordered_pts[0] = above_center[0]
        ordered_pts[1] = above_center[-1]
        ordered_pts[2] = below_center[0]
        ordered_pts[3] = below_center[-1]
        
        return ordered_pts

    

    def find_white_pixels(self):
        
        
        # find the closest point of coords to x,y
        p1 = self.find_nearest_point(self.new_points, self.corners[0])
        p2 = self.find_nearest_point(self.new_points, self.corners[1])
        p3 = self.find_nearest_point(self.new_points, self.corners[2])
        p4 = self.find_nearest_point(self.new_points, self.corners[3])
        print(p1,p2,p3,p4)
        return (p1,p2,p3,p4)

    def find_nearest_point(self,contour, point):
        """Finds the nearest point in a contour to a given point."""
        contour = np.asarray(contour)
        dists = np.sqrt(np.sum((contour - point)**2, axis=1))
        nearest_idx = np.argmin(dists)
        nearest_point = contour[nearest_idx]
        angel = 0
        for i in range(-5,5):
            idx0 = (nearest_idx + len(contour) - i -1) % len(contour)
            idx1 = (nearest_idx + len(contour) - i) % len(contour)
            idx2 = (nearest_idx + len(contour) - i + 1) % len(contour)
            m1 = (contour[idx0][1] - contour[idx1][1])/(contour[idx0][0] - contour[idx1][0])
            m2 = (contour[idx2][1] - contour[idx1][1])/(contour[idx2][0] - contour[idx1][0])
            m1 = math.atan(m1)
            m2 = math.atan(m2)
            print(f"point {contour[idx1]}, m1={m1}, m2={m2} dif={m1-m2}")

        return nearest_point 