###
# marker detection with RealSense live stream
###

## import libraries

import pyrealsense2 as rs
import numpy as np
from matplotlib import pyplot as plt 
import cv2 as cv
import sys
import os
import datetime

print("Environment ready")

### TO DO
# check versions
# pythhon 3
# opencv 4.6.0
# pyrealsense2 2.50.0
###

## function definition
def rgbChannelHistogram(channels):
    
    channel_count = np.shape(channels)[0]
    titles = "nR", "nG", "nB"
    colour = "r", "g", "b"
    bin_edges = np.linspace(0, 1, 101)
    barw = bin_edges[1]-bin_edges[0]
    
    plt.figure(figsize=(12,4))
    for c in range(channel_count):
        plt.subplot(2,channel_count,c+1)
        plt.imshow(channels[c], cmap='gray')
        plt.title('%s'%titles[c])
    
        hist_nRGB, _ =  np.histogram(channels[c], bins=bin_edges)

        plt.subplot(2,channel_count,c+4)
        plt.bar(bin_edges[:-1], hist_nRGB, width=barw, align='edge', color=colour[c])     
        plt.title('Histogram of %s'%titles[c])

        
    
def morphoTrans(frame):

    # canny edge detector
    dst_canny = cv.Canny(frame, 50, 150)
    
    # erosion/dilation,
    # kernel = np.ones((9, 9), np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
    dst_ero = cv.erode(frame, kernel, iterations=1)
    dst_dil = cv.dilate(frame, kernel, iterations=1)

    # open(e+d)/close(d+e)
    dst_open = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel, iterations=2)
    dst_close = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel, iterations=2)

    # morphological gradient (ori-e)
    dst_gradient = cv.morphologyEx(frame, cv.MORPH_GRADIENT, kernel, iterations=1)
    # tophat (ori-open)/ blackhat (ori-close)
    dst_tophat = cv.morphologyEx(frame, cv.MORPH_TOPHAT, kernel, iterations=1)
    dst_blackhat = cv.morphologyEx(frame, cv.MORPH_BLACKHAT, kernel, iterations=1)
    
    return dst_canny, dst_ero, dst_close


# def drawTrackingBox(frame, bbox):
#     x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
#     cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 3, 1)
#     cv.putText(frame, 'Tracking...', (75, 75), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)
    
# def getContours(frame, frame_contour):
    
    
    
def empty(a):
    pass




## Setup/Initialisation

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# pointclouds https://github.com/IntelRealSense/librealsense/issues/10685
# pc = rs.pointcloud() # Declare pointcloud object, for calculating pointclouds and texture mappings
# points = rs.points() # We want the points object to be persistent so we can display the last cloud when a frame drops

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

# (verify - check the auto cali example)
device_product_line = str(device.get_info(rs.camera_info.product_line))
print(device_product_line)

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)



# Getting the depth sensor's depth scale (see rs-align example for explanation)
# depth_sensor = device.query_sensors()[0]
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

# preset options
preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
current_preset = depth_sensor.get_option(rs.option.visual_preset)
# set the visual reset to high accuracy
for i in range(int(preset_range.max)):
    visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
#     print('%02d: %s'%(i,visulpreset))
    if visulpreset == "Default":
        depth_sensor.set_option(rs.option.visual_preset, i)
        current_preset = depth_sensor.get_option(rs.option.visual_preset)
        print('current preset: '+str(current_preset))    


# enable laser emitter
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1) # None, Laser, Laser auto

# enable auto exposure
depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

# # set clipping distance for background removing
# clipping_distance_in_meters = 1.2 # meter
# clipping_distance = clipping_distance_in_meters / depth_scale

# create an align object
align_to = rs.stream.color
align = rs.align(align_to)
# CHECK IF DEPTH2COLOR OT COLOR2DEPTH


config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)


print("Starting auto calibration process")


## camera Calibration

try:
    ### TO DO
    # run on-chip if necessary for health check 
    ###
    
    ### TODO
    # focal length cali 
    ###
    
    ### TO DO
    # Tare cali for improving depth measuring accuracy
    ###
    

    
    advnc_mode = rs.rs400_advanced_mode(device) #python-rs400-advanced-mode-example.py
    print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

    
finally:
    # control
    depth_sensor.set_option(rs.option.depth_units, 0.0001)

    # advanced control
    # depth control
    depth_control_get = advnc_mode.get_depth_control()
    depth_control = rs.STDepthControlGroup()
    
    depth_control.deepSeaSecondPeakThreshold = 150
    depth_control.deepSeaNeighborThreshold = 7
    depth_control.deepSeaMedianThreshold = 500
    depth_control.plusIncrement = 10
    depth_control.minusDecrement = 10
    depth_control.scoreThreshA = 1
    depth_control.scoreThreshB = 2047
    depth_control.lrAgreeThreshold = 24
#     depth_control.textureDifferenceThreshold = 65
#     depth_control.textureCountThreshold = 20
    
    advnc_mode.set_depth_control(depth_control)
    print(advnc_mode.get_depth_control())
    
    # rsm
    rsm_get = advnc_mode.get_rsm()
    rsm = rs.STRsm()
    
#     rsm.rsmBypass = 1
    rsm.diffThresh = 4
    rsm.sloRauDiffThresh = 1
    rsm.removeThresh = 63

    advnc_mode.set_rsm(rsm)
    print(advnc_mode.get_rsm())
    
    
    # depth table
    depth_table_get = advnc_mode.get_depth_table()
    depth_units = rs.STDepthTableControl()
    
    depth_units.depthClampMax = 65536
    depth_units.depthUnits = 1000
    depth_units.disparityShift = 10
    
    advnc_mode.set_depth_table(depth_units)
    print(advnc_mode.get_depth_table())

    
    
    
    ## post-processing
    # threshold filter - min&max distance
    threshold_filter = rs.threshold_filter()
    threshold_filter.set_option(rs.option.min_distance, 0.1)
    threshold_filter.set_option(rs.option.max_distance, 1.3)

    # decimation filter - default 2
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 1)

    # disparity transform
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(True)

    # Spatial filter - edge smoothing
    spatial = rs.spatial_filter()
    # emphasize the effect of smoothing filter
    spatial.set_option(rs.option.filter_magnitude, 2) # iterations
    spatial.set_option(rs.option.filter_smooth_alpha, 0.86) # exponential moving average
    spatial.set_option(rs.option.filter_smooth_delta, 20) # step-size boundary for edges preserving
    spatial.set_option(rs.option.holes_fill, 1) # 2 pixels radius

    # temporal filter - motion smoothing
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.holes_fill, 1) # persistency_index: Valid in 8/8

    # holes filling filter
    # hole_filling = rs.hole_filling_filter()
    # hole_filling.set_option(rs.option.holes_fill, 1) # farest_from_around:1, nearest_from_around:2


print("device Calibrated")   


## initialisation of detection/tracker

# tracker = cv.legacy.TrackerTLD_create()
# tracker = cv.TrackerTLD_create()

hsv_r_lb = np.array([0, 50, 60])
hsv_r_ub = np.array([20, 255, 255])
hsv_g_lb = np.array([30, 50, 0])
hsv_g_ub = np.array([100, 255, 255])
hsv_b_lb = np.array([100, 50, 0])
hsv_b_ub = np.array([170, 255, 255])


# background subtraciton for moving objects
# subtractor = cv.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)



## create pipeline - Start streaming
pipe_profile = pipeline.start(config)

curr_frame = 0


print("starting detector&tracker calibration process")


# create a new folder for images
# current_dir = os.getcwd()
# image_dir = os.path.join(current_dir, "images_"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
# if not os.path.exists(image_dir):
#     os.makedirs(image_dir)
    
depth_matrices = []


# Accessing/Visualising data for advanced calibration

while True:
    try:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        if frames.size() <2:
            # Inputs are not ready yet
            continue
    except (RuntimeError):
        print('frame count', i-1)
        pipeline.stop()
        break
            
            
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # post processing
#         # threshold filter - min&max distance
#         depth_frame = threshold_filter.process(depth_frame)
#         # decimation
#         depth_frame = decimation.process(depth_frame)
#         # disparity transform
#         depth_frame = depth_to_disparity.process(depth_frame)
#         depth_frame = disparity_to_depth.process(depth_frame)
#         #spatial/temporal filter
#         depth_frame = spatial.process(depth_frame)
#         depth_frame = temporal.process(depth_frame)
#         #hole filling filter
# #         depth_frame = hole_filling.process(depth_frame)


    # align the depth frame to color frame after post-processing
    # to avoid a distortion effect/aliasing that can cause jagged lines
    aligned_frames = align.process(frames)

    # Update color and depth frames:
    aligned_depth_frame = aligned_frames.get_depth_frame() # 640*480 depth image
    color_frame = aligned_frames.get_color_frame()
    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        continue


        
        
        
    ### TO DO    
    # Intrinsics & Extrinsics
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

#         print(depth_intrin.ppx, depth_intrin.ppy) 319.07586669921875 233.1872100830078
   ###



    # convert images to numpy arrays
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data()) # (480, 640, 3)
    
    scaled_depth_image = depth_image * depth_scale
        



    ## marker detection
    # - colour filtering
    
#     gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
#     hsv = cv.cvtColor(color_image, cv.COLOR_BGR2HSV)

#     mask_r = cv.inRange(hsv, hsv_r_lb, hsv_r_ub)
#     mask_g = cv.inRange(hsv, hsv_g_lb, hsv_g_ub)
#     mask_b = cv.inRange(hsv, hsv_b_lb, hsv_b_ub)
#     hsv_mask = mask_r + mask_g + mask_b  
#     result = cv.bitwise_and(bg_removed, bg_removed, mask=hsv_mask)

    

    # rgb channels
    B, G, R = cv.split(color_image)

    # convert to floating point
    fR = np.asarray(R, dtype=float)
    fG = np.asarray(G, dtype=float)
    fB = np.asarray(B, dtype=float)
    
    #normalised 
    my_eps = 0.001 # Avoids divide by zero!
    denominator = (fR+fG+fB+my_eps)
    nR = fR/denominator
    nG = fG/denominator
    nB = fB/denominator
    # plot histogram
#     rgbChannelHistogram([nR, nG, nB])
   

    # binary images with threshold
    mask_r = nR > 0.4 #0.43
    mask_g = nG > 0.43
    mask_b = nB > 0.47

    plt.figure(figsize=(12,3))
    plt.subplot(1,3,1)
    plt.imshow(255*mask_r, interpolation=None) #'none'
    plt.subplot(1,3,2)
    plt.imshow(255*mask_g, interpolation=None)
    plt.subplot(1,3,3)
    plt.imshow(255*mask_b, interpolation=None)
    
    # convert to uint8 format
    mask_r_int = mask_r.astype(np.uint8)*255
    mask_g_int = mask_g.astype(np.uint8)*255
    mask_b_int = mask_b.astype(np.uint8)*255

    
    
    # - contour filtering
    frame_contour = color_image.copy()  
    gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
    
    #cropped colour image
#     cv.circle(frame_contour, (80, 80), 3, (0, 0, 255), -1)
#     cv.circle(frame_contour, (80, 400), 3, (0, 0, 255), -1)
#     cv.circle(frame_contour, (560, 80), 3, (0, 0, 255), -1)
#     cv.circle(frame_contour, (560, 400), 3, (0, 0, 255), -1)

#     blurred = cv.GaussianBlur(gray, (5, 5), 0)
#     thresh = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)
#     thresh = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

     # morpholoogical transformations
#     dst_canny, dst_dil, dst_close = morphoTrans(gray)
    
   
    # find contours in the thresholded image and initialize the shape detector
    _,_,mask_r_int = morphoTrans(mask_r_int)  
    contours_r, _ = cv.findContours(mask_r_int,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for cnt in contours_r:
        
        area = cv.contourArea(cnt)
        if area > 30 and area < 2000:
        
            arc = cv.arcLength(cnt,True)
            epsilon = 0.02*arc
            approx = cv.approxPolyDP(cnt,epsilon,True)
#             hull = cv.convexHull(cnt, returnPoints = True)
#             for h in hull:
#                 h = h[0,:]
#                 cv.circle(frame_contour, h, 3, (0, 0, 255), -1)

#             cv.drawContours(frame_contour, [cnt], 0, (0, 0, 255), 1)
            
            # rectangle marker
            if len(approx) == 4:
                rect = cv.minAreaRect(cnt)
                w, h = rect[1]
                if abs(w-h) <= 5:
                    box = np.int0(cv.boxPoints(rect))
                    cv.drawContours(frame_contour,[box],0,(0,0,255),2)

                    M = cv.moments(cnt)
                    if M['m00'] != 0.0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv.circle(frame_contour, (cx, cy), 3, (0, 0, 255), -1)

                    cv.putText(frame_contour, "Rec", (cx, cy), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 0, 255), 2)
            
            # triangle marker
            elif len(approx) == 3:
                rect = cv.minAreaRect(cnt)
                w, h = rect[1]
                if abs(w-h) <= 5:
                    cv.drawContours(frame_contour,[cnt],0,(255,0,255),2)
                    M = cv.moments(cnt)
                    if M['m00'] != 0.0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv.circle(frame_contour, (cx, cy), 3, (255,0,255), -1)
                    cv.putText(frame_contour, "Tri", (cx, cy), cv.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,0,255), 2)
                
                
    _,_,mask_g_int = morphoTrans(mask_g_int)       
    contours_g, _ = cv.findContours(mask_g_int,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    # circle marker
    for cnt in contours_g:
        # simple blob 
        # hough transform
#         circles = cv.HoughCircles(mask_g_int, cv.HOUGH_GRADIENT, 1, 480 / 8,
#                         param1=100, param2=30,
#                        minRadius=3, maxRadius=100)
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             for i in circles[0, :]:
#                 center = (i[0], i[1])
#                 # circle center
#                 cv.circle(frame_contour, center, 3, (0, 255, 0), -1)
#                 cv.putText(frame_contour, "Circle", center, cv.FONT_HERSHEY_SIMPLEX,
#                     0.5, (0, 255, 0), 2)
#                 # circle outline
#                 radius = i[2]
#                 cv.circle(frame_contour, center, radius, (0, 255, 0), 2)

        # Minimum Enclosing Circle
        area = cv.contourArea(cnt)
        if area > 30 and area < 1000:
            arc = cv.arcLength(cnt,True)
            epsilon = 0.02*arc
            approx = cv.approxPolyDP(cnt,epsilon,True)
            
            x, y, w, h = cv.boundingRect(approx)
            if abs(w-h)<=5 and len(approx) >= 8:
#                 (cx,cy),radius = cv.minEnclosingCircle(cnt)
#                 cx = int(cx)
#                 cy = int(cy)
#                 cv.circle(frame_contour,(cx, cy),int(radius),(0,255,0),2)
#                 

                M = cv.moments(cnt)
                if M['m00'] != 0.0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv.rectangle(frame_contour, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv.circle(frame_contour, (cx, cy), 3, (0, 255, 0), -1)
                    cv.putText(frame_contour, "Circle", (cx, cy), cv.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
               
                

        # ellipse
#                 ellipse = cv.fitEllipse(cnt)
#                 cv.ellipse(frame_contour,ellipse,(0,255,0),2)


        
        
        

    _,_,mask_b_int = morphoTrans(mask_b_int)  
    contours_b, _ = cv.findContours(mask_b_int,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)  
    for cnt in contours_r:
        
        area = cv.contourArea(cnt)
        if area > 30 and area < 1000:
        
            arc = cv.arcLength(cnt,True)
            epsilon = 0.02*arc
            approx = cv.approxPolyDP(cnt,epsilon,True)
            # triangle marker
            if len(approx) == 3:
                rect = cv.minAreaRect(cnt)
                w, h = rect[1]
                if abs(w-h) <= 5:
                    cv.drawContours(frame_contour,[cnt],0,(255,0,255),2)
                    M = cv.moments(cnt)
                    if M['m00'] != 0.0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv.circle(frame_contour, (cx, cy), 3, (255,0,255), -1)
                    cv.putText(frame_contour, "Tri", (cx, cy), cv.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,0,255), 2)
#     # Cross marker
#     for cnt in contours_b:
        
#         area = cv.contourArea(cnt)
#         if area > 50 and area < 2000:
#             # hough line
#             # edge detection
# #             # edges = cv.Canny(gray, 150, 200)
# #             edges = cv.Canny(mask, 75, 150)

# #             #line detection
# #             lines = cv.HoughLinesP(edges, 1, np.pi/180, 60, minLineLength=15, maxLineGap=15)
# #             if lines is not None:
# #                 for line in lines:
# #                     x1, y1, x2, y2 = line[0]
# #                     cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
#             # convex hull
#             convexHull = cv.convexHull(cnt)
#             cv.drawContours(frame_contour, [convexHull], -1, (255, 0, 0), 2)

   

    # 
    ###






    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    # depth visualization 
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.visual_preset, 0) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
    colorizer.set_option(rs.option.histogram_equalization_enabled, 1)
    colorizer.set_option(rs.option.min_distance, 0.3)
    colorizer.set_option(rs.option.max_distance, 1.2)

    depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())      


    
    
    
    
    
    
    # Show images
    images = np.hstack((color_image, depth_colormap))

    
    cv.namedWindow('canny', cv.WINDOW_AUTOSIZE)
    cv.imshow('canny',mask_b_int)
    
    cv.namedWindow('contour', cv.WINDOW_AUTOSIZE)
    cv.imshow('contour',frame_contour)
    

#     cv.namedWindow('thresh', cv.WINDOW_AUTOSIZE)
#     cv.imshow('thresh',thresh)
    
    
    
    
    # save images to directory
    depth_matrices.append(scaled_depth_image)
    depth_fname = "depth"
    
    color_fname = "frame{:06d}".format(curr_frame) + ".png"
#     cv.imwrite(os.path.join(image_dir, "color_" + color_fname), color_image)
    
    



    # quit
    if cv.waitKey(1) & 0xFF == ord('q'):
    # 1ms
    # if the video is frame 30, waitKey = 1000//30, must be integer
        break


    curr_frame += 1

# finally:
    
# Stop streaming
pipeline.stop()
print("stop streaming")

# save depth matrices into .npy file
# np.save(os.path.join(image_dir, depth_fname), np.array(depth_matrices))
print("Size of depth matrices:", len(depth_matrices))
# print("\nData summary:\n", np.load('depth.npy')) (len(depth_matrices), 480, 640)

# Closes all the frames
cv.destroyAllWindows()