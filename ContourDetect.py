# -*- coding: utf-8 -*-


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math



# convert 32sC1 images to 8UC1 images for cv.
def convert_to_uint8(img):
    if np.shape(img)[2] > 1: # float64
        gray_img = img.astype(np.uint8)
        gray_img = cv.cvtColor(gray_img, cv.COLOR_HSV2BGR)
        gray_img = cv.cvtColor(gray_img, cv.COLOR_BGR2GRAY)
    else:
        pass
    
    return gray_img

# find contour based on colour masking
def find_contour(mask, img, marker_size):
    
    ppm = None

    
    cp_tri, cp_squ, cp_cro, detection, missing = detect_polygon(img, mask, marker_size)
    cp_cir, det_cir, miss_cir, ppm = detect_circle(img, mask, marker_size, ppm)   
    

    detection.insert(1, det_cir)
    missing = np.insert(missing, 1, miss_cir)
    
    
    
    return  cp_squ, cp_cir, cp_tri, cp_cro, detection, missing, ppm


# detect squares, triangles, and cross
def detect_polygon(img, mask, marker_size):
    
    cp_tri = []
    cp_squ = []
    # cp_cir = []
    cp_cro = []
    
    det_tri = 0
    det_squ = 0
    # det_cir = 0
    det_cro = 0
    
    miss_tri = 0
    miss_squ = 0
    # miss_cir = 0
    miss_cro = 0
    
    cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # sort contours based on (x,y) from left to right
    cnts = sorted(cnts, key=lambda x: [cv.boundingRect(x)[0], cv.boundingRect(x)[1]])
    # for c in cnts:
    #     print(cv.boundingRect(c)[0], cv.boundingRect(c)[1]) #x,y
        
    
    
    for cnt in cnts:
        
        area = cv.contourArea(cnt)
        
        if area > 100:
            arc = cv.arcLength(cnt,True)
            epsilon = 0.03*arc
            # Approximates a polygonal curve(s) with the specified precision
            approx = cv.approxPolyDP(cnt, epsilon, True)
            
            
            # straight bounding rectangle
            x, y, w, h = cv.boundingRect(approx)
            # rotated rectangle
            rect = cv.minAreaRect(cnt)
            box = np.int0(cv.boxPoints(rect))
            
            # image moments for centroid of contour
            M = cv.moments(cnt)
            if M['m00'] != 0.0:
                # centroids of shape
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            
            
            # triangle
            if len(approx)==3:
                # length of three sides using euclidean distance
                a = np.linalg.norm((approx[0]-approx[1]))
                b = np.linalg.norm((approx[0]-approx[2]))
                c = np.linalg.norm((approx[1]-approx[2]))
                sides = [a,b,c]

                if any([abs(sides[i-1]-sides[i])>5 for i in range(3)]):
                    pass
                
                else:
                    # average length
                    s = np.mean([a,b,c])
                
                    # pixels per metric (mm)
                    # ppm = s/marker_size
                    # print("tri: ", ppm)
                    
                    cp_tri.append([cx, cy, s])
                    
                    det_tri += 1
                    
                    cv.drawContours(img, [cnt], 0, (255,0,0), 2)
                    cv.putText(img, "Tri", (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv.circle(img, (cx, cy), 3, (255, 0, 0), -1)
                    
                
            #square
            elif len(approx)==4:
                # Euclidean distance between the midpoints of boundingbox
                d_w = np.linalg.norm((box[0]+box[3])/2-(box[1]+box[2])/2)
                d_h = np.linalg.norm((box[1]+box[0])/2-(box[2]+box[3])/2)
   
                if abs(d_w-d_h) > 5:
                    pass
                else:
                    s = np.mean([d_w, d_h])
                    
                    # pixels per metric (mm)
                    ppm = d_w/marker_size, d_h/marker_size
                    print("squ: ", ppm)
                    
                    cp_squ.append([cx, cy, s])
                    
                    
                    det_squ += 1
                    
                    cv.drawContours(img, [box], 0, (255,0,255), 2)
                    cv.putText(img, "Rec", (cx, cy), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 255), 2)
                    cv.circle(img, (cx, cy), 3, (255, 0, 255), -1)
                    
            
            # # circle - perform worse than Hough Circle
            # elif len(approx) == 8 : 
            #     s = np.mean([w,h])
            
            #     cp_cir[0].append(cx)
            #     cp_cir[1].append(cy)
            #     cp_cir[2].append(s)
            
            #     det_cir += 1
            
            #     cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            #     # cv.drawContours(img, [box], 0, (0,255,255), 2)
            #     cv.putText(img, "Cir", (cx, cy), cv.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 2)
            #     cv.circle(img, (cx, cy), 3, (0, 255, 255), -1)
            
            
            # cross
            elif len(approx) > 8:
                if abs(w-h)>5:
                    pass
                
                else:
                    s = np.mean([w,h])
                    
                    cp_cro.append([cx, cy, s])
                    
                    det_cro += 1
                    
                    # pixels per metric (mm)
                    # ppm = [w/marker_size, h/marker_size]
                    # print("cro: ", ppm)
                    
                    cv.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
                    # cv.drawContours(img, [box], 0, (255,255,0), 2)
                    cv.putText(img, "Cro", (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv.circle(img, (cx, cy), 3, (255, 255, 0), -1)
                
            else:
                pass
    
    
    
    # detection = [det_squ, det_cir, det_tri, det_cro]
    detection = [det_squ, det_tri, det_cro]
    
    missing = np.zeros(3)
    for d in range(3):
        if detection[d] == 0:
            missing[d] = 1
    
    
    return cp_tri, cp_squ, cp_cro, detection, missing


# Hough circle detection
def detect_circle(img, mask, marker_size, ppm):
    circles = cv.HoughCircles(image=mask,
                              method=cv.HOUGH_GRADIENT_ALT, # variation of HOUGH_GRADIENT to get better accuracy
                              dp=1.5,                       # Inverse ratio of the accumulator resolution to the image resolution
                              minDist=10,
                              param1=300,                   # higher threshold passing to canny edge detector
                              param2=0.7,
                              minRadius=10,
                              maxRadius=100)
    
    cp_cir = []
    
    det_cir = 0
    miss_cir = 0

    
    if circles is None:
        miss_cir = 1
        pass
        
    else:
        det_cir = 1
        circles = circles.squeeze()
       
        if circles.ndim == 2:
            cx, cy, r = circles[:,0], circles[:,1], circles[:,2]
            cx_new, cy_new, r_new = remove_outlier(circles)
            
            cx = np.mean(cx_new)
            cy = np.mean(cy_new)
            r = np.mean(r_new)
            
        else:
            cx, cy, r = circles[0], circles[1], circles[2]
            
        
        ppm = 2*r/marker_size
        # print("cir: ", ppm)
        
        cx = int(cx)
        cy = int(cy)
        r = int(r)
        
        cp_cir.append([cx, cy, r*2])
        

        cv.circle(img, (cx, cy), r, (0,255,0),2)
        cv.circle(img, (cx, cy), 2, (0,255,0),3)
        cv.putText(img, "Cir", (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    return cp_cir, det_cir, miss_cir, ppm


# remove outlier for multiple detection for a single shape
def remove_outlier(dataset):
    
    if type(dataset)=='numpy.ndarray':
        pass
    else:
        dataset = np.array(dataset)
    
    
    x, x_out = calculate_IQR(dataset[:,0])
    y, y_out = calculate_IQR(dataset[:,1])
    s, s_out = calculate_IQR(dataset[:,2])

    x_new = []
    y_new = []
    s_new = []
    
    for xi, yi, si in zip(x, y, s):
        valid_x = all([xi != x_o for x_o in x_out])
        valid_y = all([yi != y_o for y_o in y_out])
        valid_s = all([si != s_o for s_o in s_out])
        
        if valid_x and valid_y and valid_s:
            x_new.append(xi)
            y_new.append(yi)
            s_new.append(si)

    
           
    return x_new, y_new, s_new


# Percentiles calculation
def calculate_IQR(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    
    IQR = (Q3-Q1)*1.5
    
    lb = Q1 - IQR
    ub = Q3 + IQR
    
    data_IQR = []
    outliers = []
    
    for d in data:
        if d >= lb and d <= ub:
            data_IQR.append(d)
        else:
            outliers.append(d)
    
    return data_IQR, outliers



# filter the iamge and find contorus
# sort the contours from left to right
# find the corners of the first contour (us quarter)
# find the distnce between any two adjacent corners of the object in pixels e.g. 200pixel
# pixels per metric = distance between corners in pixels/distance between corners in cm e.g. 4cm
#                   =200/40mm = 5 pixels per mm
# loop and change when distance changes
# def calculate_pixels_per_metric():
    




