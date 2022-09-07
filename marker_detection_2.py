# -*- coding: utf-8 -*-

# =============================================================================
# Import libraries and modules
# =============================================================================
import os
import argparse
from sys import exit

import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import animation
import pyrealsense2 as rs

from sklearn import mixture, cluster

import xlsxwriter

import ColorDetect
import ContourDetect

cd = ColorDetect.ColorDetect()

import time

# =============================================================================
# Class definition
# =============================================================================

class RealSenseCamera:
    def __init__(self):
        
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        # device_product_line = str(device.get_info(rs.camera_info.product_line))

        # Getting the depth sensor's depth scale?
        
        # clipping distance?
        
        # preset option
        
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        

        # create pipeline
        profile = self.pipeline.start(config)
        
        
    def get_frame(self):
        # create an aign object
        # CHECK DEPTH2COLOR or COLOR2DEPTH
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        # skip the first few frames
        for _ in range(10):
            self.pipeline.wait_for_frames()
        
        centre_x = []
        centre_y = []
        detections = 0
        frame_no = 0
        
        file = open("color_points_stream.txt","w")
            
        while True:
            
            timer = cv.getTickCount()
            
            # get frameset
            frames = self.pipeline.wait_for_frames()
            
            # align depth and color frames
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            
            self.depth_frame = depth_frame
            
            # get intrinsic and extrinsic parameters
            self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            
            depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
            color_to_depth_extrin = color_frame.profile.get_extrinsics_to(depth_frame.profile)
            
            global color_image, depth_image
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            color_copy = color_image.copy()
            color_copy = cv.cvtColor(color_copy, cv.COLOR_BGR2RGB) 
            # plt.matshow(frame_contour)
            
            # gray out the background with clipping distance
            
            # rgb masks
            
            
            # contour
            
            
            
            
            
                
                
           ## depth image processing     
                
                
                
                
                
                
                
                
                
            
            # visualisation - colorise dpeth frame to colormap
            colorizer = rs.colorizer()
            colorizer.set_option(rs.option.color_scheme, 3); # white to black
            colorizer.set_option(rs.option.min_distance, 0.3)
            colorizer.set_option(rs.option.max_distance, 1.5)
            
            # Convert frames to np.array
            global depth_color_image
            depth_color_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            
            # plt.colorbar(plt.matshow(depth_image*self.depth_scale, cmap = "gray_r"))
            # plt.colorbar(plt.matshow(depth_color_image, cmap = "gray"))

            
            #fps
            fps = cv.getTickFrequency()/(cv.getTickCount()-timer)
            cv.putText(color_copy, 'FPS: ' + str(int(fps)), (75, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 1)
            
            
            
            color_copy = cv.cvtColor(color_copy, cv.COLOR_RGB2BGR)            
            images = np.hstack((color_copy, depth_color_image))
            
            # show images in opencv window
            show_image(images)

            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                self.release()
                break
            
        
    def release(self):
        self.pipeline.stop()



class ROSBagFile:
    def __init__(self, inputbag=None):
        
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(inputbag, False)
        config.enable_all_streams()

        # create pipeline
        pipeline_profile = self.pipeline.start(config)
        device = pipeline_profile.get_device()
        
        # depth scale "depth unit" 
        #  - translates pixels to meters: 0.0001 metre
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(self.depth_scale)
        
        # clipping distance
        # clipping_distance_in_meters = 1.5 # meter
        # self.clipping_distance = clipping_distance_in_meters/self.depth_scale

        playback = device.as_playback()
        playback.set_real_time(False)
        
        

    def get_frame(self):
        # create an aign object
        # CHECK DEPTH2COLOR or COLOR2DEPTH
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        # skip the first few frames
        for _ in range(10):
            self.pipeline.wait_for_frames()
            
            

        # capture a sequence with 60 frames
        frame_length = 60
        queue = rs.frame_queue(frame_length)
        
        
        for i in range(frame_length):
            
            # get frame from pipeline
            frames = self.pipeline.wait_for_frames()
            frame_number = frames.frame_number
            
            frames.keep()
            queue.enqueue(frames)
            
            aligned_frames = align.process(frames)
            
        self.pipeline.stop() 
        
        

        # initialisation of some vvariables
        marker_size = 40 # 40mm
    
        H = 720 # 720
        W = 1280 # 1280
        
        x1 = int(np.round(0.3*W))
        y1 = int(np.round(0.15*H))
        w = int(np.round(0.4*W))
        h = int(np.round(0.5*H))
        roi = initialise_roi(x1, y1, w, h)
        
        [x1, y1, w, h] = roi
        
        
        
        frame_no = 0
        
        detections = np.zeros((3,4)).tolist()
        missings = np.zeros((3,4)).tolist()
        colour_correct = np.zeros((3,4)).tolist()
        colour_error = np.zeros((3,4)).tolist()
        shape_correct = np.zeros((3,4)).tolist()
        shape_error = np.zeros((3,4)).tolist()
        
        depth_r_squ = []
        depth_r_cir = []
        depth_r_tri = []
        depth_r_cro = []
        
        depth_g_squ = []
        depth_g_cir = []
        depth_g_tri = []
        depth_g_cro = []
        
        depth_b_squ = []
        depth_b_cir = []
        depth_b_tri = []
        depth_b_cro = []


                
        file = open("colour_shape_data.txt","w")
        
        # 
        for i in range(frame_length):
            timer = cv.getTickCount()
            
            frames = queue.wait_for_frame()
            frame_no = frames.frame_number
            print("frame: ", i+1)
        
            # color_frame = frames.as_frameset().get_color_frame()
            color_frame = aligned_frames.as_frameset().get_color_frame()
            depth_frame = aligned_frames.as_frameset().get_depth_frame()


            self.depth_frame = depth_frame
            
            # get intrinsic and extrinsic parameters
            self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            
            depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
            color_to_depth_extrin = color_frame.profile.get_extrinsics_to(depth_frame.profile)
            


            global color_image, depth_image
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            ## colour image processing
            color_copy = color_image.copy()
            
            # resize roi image
            resize_color = color_copy[y1:y1+h, x1:x1+w, :]
            
            # GaussianBlur
            resize_color = cv.GaussianBlur(resize_color,(11,11),0)
            plt.figure(figsize=(8, 6), dpi=80)
            plt.imshow(resize_color)
            plt.axis("off")
            
            # rgb histogram of roi
            # cd.rgb_histogram(resize_color)
            
            
            # manual rgb masks
            # img_rgb = cd.rgb_masking(resize_color)
            # cd.scatter_2d(resize_color,stepsize=100)
            
            # manual hsv masks
            img_hsv, mask_r, mask_g, mask_b = cd.hsv_masking(resize_color)
            cd.scatter_2d(img_hsv,stepsize=1)
            
            
            # rgb clustering and masking
            img_hsv_2 = img_hsv.copy()
            img_hsv_2 = cv.cvtColor(img_hsv_2, cv.COLOR_BGR2HSV_FULL)
            

            flatten_data = img_hsv_2.reshape((-1,3))
            flatten_data_idx = np.array(np.where((flatten_data[:,0]!=0) & (flatten_data[:,1]!=0) & (flatten_data[:,2]!=0))).flatten()
            flatten_data = flatten_data[(flatten_data[:,0]!=0) & (flatten_data[:,1]!=0) & (flatten_data[:,2]!=0)]
            
            # clustering three channels
            
            # # cd.rgb_kmeans(np.array(data),3)
            # # cd.rgb_GMM(np.array(data),3)
            cluster_r, cluster_g, cluster_b, idx_r, idx_g, idx_b, centroids = cd.rgb_DBSCAN(np.array(flatten_data), flatten_data_idx, 3)
            
            
            # convert colour values back to pixel coordinates for masks
            mask_r = cd.get_mask(color_copy, roi, cluster_r, idx_r)
            mask_g = cd.get_mask(color_copy, roi, cluster_g, idx_g)
            mask_b = cd.get_mask(color_copy, roi, cluster_b, idx_b)
            
            
            # # find contour
            mask_r = ContourDetect.convert_to_uint8(mask_r)
            mask_g = ContourDetect.convert_to_uint8(mask_g)
            mask_b = ContourDetect.convert_to_uint8(mask_b)
            
            
            cp_squ_r, cp_cir_r, cp_tri_r, cp_cro_r, detection_r, missing_r, ppm_r = ContourDetect.find_contour(mask_r, color_copy, marker_size)
            cp_squ_g, cp_cir_g, cp_tri_g, cp_cro_g, detection_g, missing_g, ppm_g = ContourDetect.find_contour(mask_g, color_copy, marker_size)
            cp_squ_b, cp_cir_b, cp_tri_b, cp_cro_b, detection_b, missing_b, ppm_b = ContourDetect.find_contour(mask_b, color_copy, marker_size)
            
            print(ppm_r,ppm_g,ppm_b)
            # plt.matshow(color_copy)
            
            detection_new = [detection_r, detection_g, detection_b]
            detections = np.array(detections)+detection_new
            
            missing_new = [missing_r, missing_g, missing_b]
            missings = np.array(missings)+missing_new
            
            # print(detections)
            # print(missings)
            
            
            ## Detection accuracy check
            color_copy_hsv = color_image.copy()
            color_copy_hsv = cv.cvtColor(color_copy_hsv, cv.COLOR_BGR2HSV_FULL)     
            
            shape_r = np.array([cp_squ_r, cp_cir_r, cp_tri_r, cp_cro_r], dtype=object)
            shape_g = np.array([cp_squ_g, cp_cir_g, cp_tri_g, cp_cro_g], dtype=object)
            shape_b = np.array([cp_squ_b, cp_cir_b, cp_tri_b, cp_cro_b], dtype=object)
            
            shapes = [shape_r, shape_g, shape_b]
            
            
            # check colour and shape
            colour_correct_new = np.zeros((3,4))
            colour_error_new = np.zeros((3,4))
            px_percentage = np.zeros((3,4))
            
            for c in range(3):
                for sh in range(4):
                    if shapes[c][sh]is None:
                        pass
                    else:
                        colour_error_new[c,sh], px_percentage[c,sh] = check_colour_shape(color_copy_hsv, shapes[c][sh], centroids, c, 100)
                    
                        # if px_percentage[c,sh]
                        
            colour_correct_new =  detection_new-colour_error_new
            
            colour_correct = np.array(colour_correct)+colour_correct_new
            colour_error = np.array(colour_error)+colour_error_new
            
            
            # print(px_percentage)
                
            
            ## depth image processing
            
            
            # for c in range(3):
            #     for sh in range(4):
            #         if shapes[c][sh]is None:
            #             pass
            #         else:
            #             depth_points = get_depth_and_point(self.depth_frame, shapes[c][sh], self.color_intrin)
            #             print(depth_points)
                        
             
            depth_r_squ += [get_depth_and_point(self.depth_frame, shapes[0][0], self.color_intrin)]
            depth_r_cir += [get_depth_and_point(self.depth_frame, shapes[0][1], self.color_intrin)]
            depth_r_tri += [get_depth_and_point(self.depth_frame, shapes[0][2], self.color_intrin)]
            depth_r_cro += [get_depth_and_point(self.depth_frame, shapes[0][3], self.color_intrin)]
             
            depth_g_squ += [get_depth_and_point(self.depth_frame, shapes[1][0], self.color_intrin)]
            depth_g_cir += [get_depth_and_point(self.depth_frame, shapes[1][1], self.color_intrin)]
            depth_g_tri += [get_depth_and_point(self.depth_frame, shapes[1][2], self.color_intrin)]
            depth_g_cro += [get_depth_and_point(self.depth_frame, shapes[1][3], self.color_intrin)]
            
            depth_b_squ += [get_depth_and_point(self.depth_frame, shapes[2][0], self.color_intrin)]
            depth_b_cir += [get_depth_and_point(self.depth_frame, shapes[2][1], self.color_intrin)]
            depth_b_tri += [get_depth_and_point(self.depth_frame, shapes[2][2], self.color_intrin)]
            depth_b_cro += [get_depth_and_point(self.depth_frame, shapes[2][3], self.color_intrin)]
            

            # visualisation - colorise dpeth frame to colormap
            colorizer = rs.colorizer()
            colorizer.set_option(rs.option.color_scheme, 3); # white to black
            colorizer.set_option(rs.option.min_distance, 0.3)
            colorizer.set_option(rs.option.max_distance, 1.5)
            
            # Convert frames to np.array
            global depth_color_image
            depth_color_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            
            # plt.colorbar(plt.matshow(depth_image*self.depth_scale, cmap = "gray_r"))
            # plt.colorbar(plt.matshow(depth_color_image, cmap = "gray"))

            


            # write into file

            file.write("\n\n frame no.: "+str(frame_no)+\
                        "\n detections: "+str(detections)+\
                        "\n missing: "+str(missings)+\
                        "\n colour correct: "+str(colour_correct)+\
                        "\n colour_error: "+str(colour_error)+\
                        "\n pixel percentage: "+str(px_percentage)
                        
                        )
                

            #fps
            fps = cv.getTickFrequency()/(cv.getTickCount()-timer)
            cv.rectangle(color_copy, (60, 30), (160,80), (255,255,255), -1)
            # cv.putText(color_copy, 'FPS: ' + str(int(fps)), (75, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 1)
            cv.putText(color_copy, str(frame_no)+' ('+str(i+1)+')', (75, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0), 1)
            
            
            color_copy = cv.cvtColor(color_copy, cv.COLOR_RGB2BGR)            
            images = np.hstack((color_copy, depth_color_image))
            
            # show images in opencv window
            show_image(images)
            
            
            time.sleep(1)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                file.close()
                break
        
       
        n_offset = np.zeros((3,4))
        
        z_r_squ, n_offset[0,0] = remove_outlier(depth_r_squ)
        z_r_cir, n_offset[0,1] = remove_outlier(depth_r_cir)
        z_r_tri, n_offset[0,2] = remove_outlier(depth_r_tri)
        z_r_cro, n_offset[0,3] = remove_outlier(depth_r_cro)
        
        z_g_squ, n_offset[1,0] = remove_outlier(depth_g_squ)
        z_g_cir, n_offset[1,1] = remove_outlier(depth_g_cir)
        z_g_tri, n_offset[1,2] = remove_outlier(depth_g_tri)
        z_g_cro, n_offset[1,3] = remove_outlier(depth_g_cro)
        
        z_b_squ, n_offset[2,0] = remove_outlier(depth_b_squ)
        z_b_cir, n_offset[2,1] = remove_outlier(depth_b_cir)
        z_b_tri, n_offset[2,2] = remove_outlier(depth_b_tri)
        z_b_cro, n_offset[2,3] = remove_outlier(depth_b_cro)
            
        points_3d = [z_r_squ, z_r_cir, z_r_tri, z_r_cro,
                     z_g_squ, z_g_cir, z_g_tri, z_g_cro,
                     z_b_squ, z_b_cir, z_b_tri, z_b_cro,]
        

        print(n_offset)
        print(points_3d)
        
            
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            file.close()
            # workbook.close()
 



# =============================================================================
# Functions definition
# =============================================================================
# load example ROS bag  
def default_bag_input():
    #Get the current working directory
    cwd = os.getcwd()
    cwd = cwd.replace(os.sep,"/")
    # inputbag = cwd + "/bags/pre_session_cali_1270_720_surface.bag"
    inputbag = "C:/Users/yuijo/OneDrive - Imperial College London/21-22/MSc Project/realsense/final data acquisition/markers/distances/single_marker_on_d0.bag"
    
    return inputbag


# Rescale image to fit window size
def rescale_image_by_ratio(image,rescale_percentage=0.5):
    resized_h = int(image.shape[0]*rescale_percentage)
    resized_w = int(image.shape[1]*rescale_percentage)
    resize_dim = (resized_w, resized_h)
    resized_image = cv.resize(image, resize_dim)
    
    return resized_image


# initial roi
def initialise_roi( x1, y1, w, h):
    roi = [x1, y1, w, h]

    return roi


# set a new roi
def set_new_roi():
    pass        
 
    
# Show images
def show_image(image,rescale_percentage=0.5):
    if image.shape[1] > 1280 or image.shape[0] > 720:
        image = rescale_image_by_ratio(image,rescale_percentage)
        
    cv.namedWindow("Stream", cv.WINDOW_AUTOSIZE)
    cv.imshow("Stream", image)
            

### -----Preset and post processing-----
# Preset options setting: only available for real-time streaming
def preset_options(device):
    depth_sensor = device.query_sensors()[0]
    color_sensor = device.query_sensors()[1]
    
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
    
    
    advnc_mode = rs.rs400_advanced_mode(device)
    print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")
    
    # enable laser emitter
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1) # None, Laser, Laser auto

    depth_sensor.set_option(rs.option.enable_auto_exposure, 1) 

    # depth table
    depth_table_get = advnc_mode.get_depth_table()
    depth_units = rs.STDepthTableControl()
    
    depth_units.depthClampMax = 65536
    depth_units.depthUnits = 1000
    depth_units.disparityShift = 10
    
    advnc_mode.set_depth_table(depth_units)
    print(advnc_mode.get_depth_table())


# Post processing filtering
def post_processing(depth_frame):
    spat_filter = rs.spatial_filter()       # Spatial    - edge-preserving spatial smoothing
    # temp_filter = rs.temporal_filter()      # Temporal   - reduces temporal noise
    hole_fill = rs.hole_filling_filter()    # Hole-filling filter
    
    filtered = spat_filter.process(depth_frame)
    # filtered = temp_filter.process(filtered)
    depth_frame_filtered = hole_fill.process(filtered)

    return depth_frame_filtered

### -----Color detection and filtering-----

# import as a seperate class from ColorDetect.py


### -----Contour detection-----

# import as a seperate module from ContourDetect.py


# Morphological transformation
def morpho_trans(frame, kernalsize=(9,9)):

    # canny edge detector
    # dst_canny = cv.Canny(frame, 50, 150)
    
    # erosion/dilation,
    # kernel = np.ones((9, 9), np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernalsize)
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
    
    return  dst_dil


### ---- 2D Pixels to 3D Points -----
def get_depth_and_point(depth_frame, dataset, color_intrin):
    dataset = np.array(dataset).astype(int)
    
    depth_points = []
    
    for [x,y,_] in dataset:
        depth = depth_frame.get_distance(x, y) # depth data from depth frame
        depth_point = rs.rs2_deproject_pixel_to_point(color_intrin, (x, y), depth)
        depth_points.append(depth_point)
        # range2point = math.sqrt(((color_point[0])**2) + ((color_point[1])**2) + ((color_point[2])**2))
        
        # print('| 2D '+ str(x) + ',' + str(y) + ' --> 3D ' + str(depth_point))
    
    return np.mean(depth_points, axis=0)    


### -----Outliler removal ----          
# remove outlier for centre points determining
def remove_outlier(dataset):
    
    if type(dataset)=='numpy.ndarray':
        pass
    else:
        dataset = np.array(dataset)
    
    if np.isnan(dataset).all():
        xyz_new, offset_no = [], 0
    else: 
        x, x_out = calculate_IQR(dataset[:,0])
        y, y_out = calculate_IQR(dataset[:,1])
        z, z_out = calculate_IQR(dataset[:,2])
    
        x_new = []
        y_new = []
        z_new = []
        
        for xi, yi, zi in zip(x, y, z):
            valid_x = all([xi != x_o for x_o in x_out])
            valid_y = all([yi != y_o for y_o in y_out])
            valid_z = all([zi != z_o for z_o in z_out])
            
            if valid_x and valid_y and valid_z:
                x_new.append(xi)
                y_new.append(yi)
                z_new.append(zi)
    
        xyz_new = np.mean([x_new, y_new, z_new],axis=1)
        
        offset_no = len(x)-len(x_new)
    
    return xyz_new, offset_no


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


def check_colour_shape(img, shape, centroids, colour, threshold=100):
    error = 0
    px_percentage = 0
    for det in shape:
        cx,cy,s = det[0],det[1],det[2]
        
        ty = int(cy-s/2)
        by = int(cy+s/2)
        lx = int(cx-s/2)
        rx = int(cx+s/2)
        
        mid = int(s/2)
        
        det_roi = img[ty:by,lx:rx :]
        
        d = np.zeros(3)
        
        # calculate euclidean distance
        for c in range(3):
            d[c] = np.linalg.norm(np.array(det_roi[mid,mid,:])-np.array(centroids[c]))#,axis=1
        if (np.argsort(d)[0] != colour or min(d) > threshold):
            error += 1
        else:
            n_colour_px = np.sum([det_roi[:,:,1]>100])#
            # print(n_colour_px)
            n_total_px = s**2
            # print(n_total_px)
            px_percentage = n_colour_px/n_total_px
            # print(colour, px_percentage)
            
        
        # fig = plt.figure()
        # plt.imshow(cv.cvtColor(det_roi,cv.COLOR_HSV2BGR_FULL))#
        # plt.show()
    
    return error, px_percentage


# =============================================================================
# Main code
# =============================================================================
if __name__ == '__main__':
    ### TO DO
    # check versions
    # phthon 3, cv 4.6, rs 2.50
    
    parser = argparse.ArgumentParser(description="Access Intel RealSense camera, or input a .bag file")
    # Add arguments
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
    # Parse the command line arguments to an object
    args = parser.parse_args()
    
    # Check arguments
    if not args.input:
        # check if camera connected
        # device info: device name, info: name, firmware version, usb type
        ctx = rs.context()
        if len(ctx.devices) > 0:
            for d in ctx.devices:
                print ('Found device:\n', \
                           'Camera name:', d.get_info(rs.camera_info.name), '\n', \
                           'Firmware ver.:', d.get_info(rs.camera_info.firmware_version), '\n', \
                           'USB port:', d.get_info(rs.camera_info.usb_type_descriptor))
            
            RealSenseCamera().get_frame()
            
            
        else:
            print("No Intel Device connected")
            print("Start using example bag")
            # default bag for testing
            inputfile = default_bag_input()
            ROSBagFile(inputfile).get_frame()
    
        
    # Check if the given file have bag extension
    elif os.path.splitext(args.input)[1] != ".bag":
        print("Wrong file format.")
        print("Only .bag files are accepted")
        exit()
    
    else:
        inputfile = args.input
        ROSBagFile(inputfile).get_frame()
    
    







