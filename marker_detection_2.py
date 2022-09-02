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
           
        marker_size = 40 # 40mm
    
        H = 720 # 720
        W = 1280 # 1280
        
        
        x1 = int(np.round(0.25*W))
        y1 = int(np.round(0.15*H))
        w = 768
        h = 504
        roi = initialise_roi(x1, y1, w, h)
        
        [x1, y1, w, h] = roi
        
        
        squ_r, cir_r, tri_r, cro_r = [],[],[],[]
        squ_g, cir_g, tri_g, cro_g = [],[],[],[]
        squ_b, cir_b, tri_b, cro_b = [],[],[],[]
        
        
        frame_no = 0
        
        TPs = np.zeros((3,4)).astype(int)
        FPs = np.zeros((3,4)).astype(int)
        
        
        
        cv.namedWindow("Stream", cv.WINDOW_AUTOSIZE)
        
        file = open("accuracy_data_bag.txt","w")
        
        while True:
            timer = cv.getTickCount()
            
            # for-loop for frame sampling accumulation 
            # frame_count = 5
            # frameset_depth = np.zeros(())
            
            # get frameset
            frames = self.pipeline.wait_for_frames()
            frame_no += 1
            print("frame: ",frame_no)
            
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
            
            
            
            ## colour image processing
            color_copy = color_image.copy()
            
            # resize roi image
            resize_color = color_copy[y1:y1+h, x1:x1+w, :]
            plt.matshow(resize_color)
            
            # rgb 3d scatter plot
            filtered_points, points_idx = cd.rgb_scatter_plotting(resize_color,stepsize=1)
            
            # rgb clustering and masking
            cluster_r, cluster_g, cluster_b, idx_r, idx_g, idx_b = cd.rgb_kmeans(filtered_points,points_idx, k=3)
            # rgb_GMM(color_copy, n_components=10, p_thershold=0.95)
            
            # hsv scatter
            # filtered_points = cd.hsv_scatter_potting(resize_color,stepsize=100)
            # cd.hsv_kmeans(filtered_points, k=3)
            
            
            
            mask_r = cd.seperate_rgb_masks(H, W, resize_color, roi, cluster_r, idx_r)
            mask_g = cd.seperate_rgb_masks(H, W, resize_color, roi, cluster_g, idx_g)
            mask_b = cd.seperate_rgb_masks(H, W, resize_color, roi, cluster_b, idx_b)
            # plt.matshow(mask_r)
            # plt.matshow(mask_g)
            # plt.matshow(mask_b)
           
            
            # find contour
            mask_r = ContourDetect.convert_to_uint8(mask_r)
            mask_g = ContourDetect.convert_to_uint8(mask_g)
            mask_b = ContourDetect.convert_to_uint8(mask_b)
            
            cp_squ_r, cp_cir_r, cp_tri_r, cp_cro_r, detect_r, FP_r = ContourDetect.find_contour(mask_r, color_copy, marker_size)
            cp_squ_g, cp_cir_g, cp_tri_g, cp_cro_g, detect_g, FP_g = ContourDetect.find_contour(mask_g, color_copy, marker_size)
            cp_squ_b, cp_cir_b, cp_tri_b, cp_cro_b, detect_b, FP_b = ContourDetect.find_contour(mask_b, color_copy, marker_size)

            # plt.matshow(color_copy)
            
            if cp_squ_r:
                squ_r.extend(cp_squ_r)
            if cp_cir_r:
                cir_r.extend(cp_cir_r)
            if cp_tri_r:
                tri_r.extend(cp_tri_r)
            if cp_cro_r:
                cro_r.extend(cp_cro_r)
            
            if cp_squ_g:
                squ_g.extend(cp_squ_g)
            if cp_cir_g:
                cir_g.extend(cp_cir_g)
            if cp_tri_g:
                tri_g.extend(cp_tri_g)
            if cp_cro_g:
                cro_g.extend(cp_cro_g)
                
            if cp_squ_b:
                squ_r.extend(cp_squ_b)
            if cp_cir_b:
                cir_r.extend(cp_cir_b)
            if cp_tri_b:
                tri_r.extend(cp_tri_b)
            if cp_cro_b:
                cro_r.extend(cp_cro_b)
            
            # # detect "offset"
            # cx_squ_r, cy_squ_r, s_squ_r, offset_no_squ_r = remove_outlier(squ_r)
            # remove_outlier(cir_r)
            # remove_outlier(tri_r)
            # remove_outlier(squ_r)
            
            
            # ## depth image processing
            # depth_point = get_depth_and_point(self.depth_frame, [cx_squ_r, cy_squ_r], self.color_intrin)
            # print(depth_point)
            
            #ppm
            
            #IoU
            
            
            ## Accuracy 
            # True positive
            new_detection = np.array([detect_r, detect_g, detect_b])
            TPs = TPs + new_detection
            
            # False positive
            new_FP = np.array([FP_r, FP_g, FP_b])
            FPs = FPs + new_FP
            # print(FPs)
                
            # False negative
            FNs = (np.ones((3,4)).astype(int))*frame_no-TPs
            
            #True negative
            TNs = np.zeros((3,4)).astype(int)
            
            
            # output accuracy
            
            np.seterr(invalid='ignore')
            
            precision = TPs/(TPs+FPs)
            recall = TPs/(TPs+FNs)
            F1score = (precision*recall)/((precision+recall)/2)
            
            precision[np.isnan(precision)] = 0
            recall[np.isnan(recall)] = 0
            F1score[np.isnan(F1score)] = 0
            
            
            
            
            # morpho
            # dst_dil = morpho_trans(depth_image, (9,9))
            # plt.matshow(dst_dil, cmap = 'gray')
   
            # reset roi (AFTER THE DETECTION)
            
            
            
            
            
            
            # write into file
            # file.write("\n\n frame no.: "+str(frame_no)+\
            #             "\n detected: "+str(new_detect)+\
            #             "   %detect: "+str(perc_detect)+\
            #             "\n centre estimation:"+str([mu_x, mu_y])+\
            #             "\n 3D point: "+str(mu_3d))#+\
            #             # "\n accuracy: "+str(accuracy))
            
            workbook = xlsxwriter.Workbook('accuracy_single_marker.xlsx')
            worksheet = workbook.add_worksheet()
            
            colours = ["Red", "Green", "Blue"]
            shapes = ["Square", "Circle", "Triangle", "Cross"]
            accuracy = ["TP", "FP", "Precision", "Recall", "F1 score"]
            
            worksheet.write(0, 0, frame_no)
            
            row = 2
            cols = [1,6,11,16]
            for i in range(4):
                worksheet.write(0, cols[i], shapes[i])
                
                col = cols[i]
                for j in range(5):
                    worksheet.write(1, col+j , accuracy[j])
                for n in range(3):
                    worksheet.write(row+n, col , TPs[n,i])
                    worksheet.write(row+n, col+1 , FPs[n,i])
                    worksheet.write(row+n, col+2 , precision[n,i])
                    worksheet.write(row+n, col+3 , recall[n,i])
                    worksheet.write(row+n, col+4 , F1score[n,i])

            row = 2
            col = 0
            for c in colours:
                worksheet.write(row, col , c)
                row += 1
            
            
            
            
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
                # file.close()
                workbook.close()
                break





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
    
    
    dataset = np.array(dataset).T.astype(int)
    print(dataset)
    
    depth_points = []
    
    for [x,y] in dataset:
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
    
    offset_no = len(x)-len(x_new)
    
    return x_new, y_new, s_new, offset_no


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
    
    







