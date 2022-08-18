# -*- coding: utf-8 -*-

import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


import os
import argparse
from sys import exit


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

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(self.depth_scale)
        
        clipping_distance_in_meters = 1.2 #1 meter
        self.clipping_distance = clipping_distance_in_meters/self.depth_scale

        
        preset_options(device)

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
            
            
        while True:
            
            timer = cv.getTickCount()
            
            # get frameset
            frames = self.pipeline.wait_for_frames()
            
            # align depth and color frames
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            
            self.depth_frame = depth_frame

            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            frame_contour = color_image.copy()

            
            
            # gray out the background! to avoid detecting other place            
            gray_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            frame_contour = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), gray_color, color_image)
            

            
            frame_contour = cv.cvtColor(frame_contour, cv.COLOR_RGB2BGR) 
            # plt.matshow(frame_contour)
            mask_r, mask_g, mask_b = rgb_masking(frame_contour)
            
            
            contours_r, _ = cv.findContours(mask_r,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

            # loop over the contours
            for cnt in contours_r:
                
                area = cv.contourArea(cnt)
                if area > 30 and area < 3000:
                
                    arc = cv.arcLength(cnt,True)
                    epsilon = 0.02*arc
                    approx = cv.approxPolyDP(cnt,epsilon,True)
                    
                    # rectangle marker
                    if len(approx) == 4:
                        rect = cv.minAreaRect(cnt)
                        w, h = rect[1]
                        if abs(w-h) <= 5:
                            box = np.int0(cv.boxPoints(rect))
                            cv.drawContours(frame_contour,[box],0,(255,0,0),2)
        
                            M = cv.moments(cnt)
                            if M['m00'] != 0.0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                cv.circle(frame_contour, (cx, cy), 3, (255, 0, 0), -1)
        
                            cv.putText(frame_contour, "Rec", (cx, cy), cv.FONT_HERSHEY_SIMPLEX,
                                   0.5, (255, 0, 0), 2)
            
            contours_g, _ = cv.findContours(mask_g,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

            # circle marker
            for cnt in contours_g:
                # Minimum Enclosing Circle
                area = cv.contourArea(cnt)
                if area > 30 and area < 3000:
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
                            
            contours_b, _ = cv.findContours(mask_b,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)  
            for cnt in contours_r:
                
                area = cv.contourArea(cnt)
                if area > 10 and area < 3000:
                
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
            
            # post-processing
            # visualisation
    
            
            # colorise dpeth frame to colormap
            depth_color_frame = rs.colorizer().colorize(depth_frame)
    
            # Convert frames to np.array
            depth_color_image = np.asanyarray(depth_color_frame.get_data())
            # color_image = np.asanyarray(color_frame.get_data())
            # color_cvt = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)
                
            
            # get color intrinsic parameters
            # self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            
            
            # post_processing
            
            

             #fps
            fps = cv.getTickFrequency()/(cv.getTickCount()-timer)
            cv.putText(frame_contour, 'FPS: ' + str(int(fps)), (75, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 1)
            
            
            # plt.matshow(frame_contour)
            frame_contour = cv.cvtColor(frame_contour, cv.COLOR_RGB2BGR)            
            images = np.hstack((frame_contour, depth_color_image))
            
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
        
        
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(self.depth_scale)
        
        clipping_distance_in_meters = 1.2 #1 meter
        self.clipping_distance = clipping_distance_in_meters/self.depth_scale

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
           
        
        centre_x = []
        centre_y = []
        detected = 0
        frame_no = 0
        
        file = open("accuracy_data.txt","w")
        
        while True:
            timer = cv.getTickCount()
            
            # for-loop for frame sampling accumulation 
            # frame_count = 5
            # frameset_depth = np.zeros(())
            
            # get frameset
            frames = self.pipeline.wait_for_frames()
            frame_no += 1
            
            # align depth and color frames
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            
            self.depth_frame = depth_frame

            
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            frame_contour = color_image.copy()

            
            # gray out the background! to avoid detecting other place            
            gray_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            frame_contour = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), gray_color, color_image)
            

            mask_r, mask_g, mask_b = rgb_masking(frame_contour)
                
            
            
            
            
            contours_r, _ = cv.findContours(mask_r,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

            # loop over the contours
            for cnt in contours_r:
                
                area = cv.contourArea(cnt)
                if area > 30 and area < 3000:
                
                    arc = cv.arcLength(cnt,True)
                    epsilon = 0.02*arc
                    approx = cv.approxPolyDP(cnt,epsilon,True)
                    
                    # rectangle marker
                    if len(approx) == 4:
                        rect = cv.minAreaRect(cnt)
                        w, h = rect[1]
                        if abs(w-h) <= 5:
                            box = np.int0(cv.boxPoints(rect))
                            cv.drawContours(frame_contour,[box],0,(255,0,0),2)
        
                            M = cv.moments(cnt)
                            if M['m00'] != 0.0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                cv.circle(frame_contour, (cx, cy), 3, (255, 0, 0), -1)
                                
                                centre_x.append(cx)
                                centre_y.append(cy)
                                detected += 1
                                
        
                            cv.putText(frame_contour, "Rec", (cx, cy), cv.FONT_HERSHEY_SIMPLEX,
                                   0.5, (255, 0, 0), 2)
            
            contours_g, _ = cv.findContours(mask_g,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

            # circle marker
            for cnt in contours_g:
                # Minimum Enclosing Circle
                area = cv.contourArea(cnt)
                if area > 30 and area < 3000:
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
                            
                            centre_x.append(cx)
                            centre_y.append(cy)
                            detected += 1
                            
                            cv.rectangle(frame_contour, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv.circle(frame_contour, (cx, cy), 3, (0, 255, 0), -1)
                            cv.putText(frame_contour, "Circle", (cx, cy), cv.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
                            
            contours_b, _ = cv.findContours(mask_b,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)  
            for cnt in contours_r:
                
                area = cv.contourArea(cnt)
                if area > 30 and area < 3000:
                
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
                                
                                centre_x.append(cx)
                                centre_y.append(cy)
                                detected += 1
                                
                                cv.circle(frame_contour, (cx, cy), 3, (255,0,255), -1)
                            cv.putText(frame_contour, "Tri", (cx, cy), cv.FONT_HERSHEY_SIMPLEX,
                            0.5, (255,0,255), 2)
            
            # post-processing
            # visualisation
        
            
            
            if not centre_x or not centre_y:
                pass
            else:
                max_d = 5
                
                mu_x = np.mean(centre_x)
                mu_y = np.mean(centre_y)
                
                sd_x = np.std(centre_x)
                sd_y = np.std(centre_y)
                
                print(mu_x, mu_y)
                
            no_outlier_x = []
            no_outlier_y = []
            for c in range(np.shape(centre_x)[0]):
                if centre_x[c] > (mu_x-max_d*sd_x) and centre_x[c] < (mu_x+max_d*sd_x) and centre_y[c] > (mu_y-max_d*sd_y) and centre_y[c] < (mu_y+max_d*sd_y):    
                    no_outlier_x.append(centre_x[c])
                    no_outlier_y.append(centre_y[c])
                
                
            if not no_outlier_x or not no_outlier_y:
                print("no detection")
            else:
                
                new_mu_x = np.mean(no_outlier_x)
                new_mu_y = np.mean(no_outlier_y)
                
                # accuracy = 
                
                perc_detect = (frame_no*3-detected)/(frame_no*3)
                
                print(new_mu_x, new_mu_y)
                print(frame_no, detected, perc_detect)
                
                
                
                # total_no_pixel = np.shape(color_image)[0]*np.shape(color_image)[1]
                # print(np.sum(R>0.4))# count colored pixels 
                # print(cv.countNonZero(mask_r_int))#find specific region
                
                file.write("\n\n frame no.: "+str(frame_no)+\
                            "\n detected: "+str(detected)+\
                            "   %detect: "+str(perc_detect)+\
                            "\n centre_before:"+str([mu_x, mu_y])+\
                            "\n centre_after: "+str([new_mu_x, new_mu_y]))#+\
                            # "\n accuracy: "+str(accuracy))

            
            # colorise dpeth frame to colormap
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            # Convert frames to np.array
            depth_color_image = np.asanyarray(depth_color_frame.get_data())
            # scaled_depth_image = depth_color_image * self.depth_scale
            
            # plt.matshow(depth_color_image)
            # plt.title('Using wait without for-loop accumulation')
            # plt.savefig('op1.png')
            
            
            # color_image = np.asanyarray(color_frame.get_data())
            # frame_contour = cv.cvtColor(frame_contour, cv.COLOR_BGR2RGB)
            
            
            # get color intrinsic parameters
            self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            
             #fps
            fps = cv.getTickFrequency()/(cv.getTickCount()-timer)
            cv.putText(frame_contour, 'FPS: ' + str(int(fps)), (75, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 1)
            
            
            # plt.matshow(frame_contour)
            frame_contour = cv.cvtColor(frame_contour, cv.COLOR_RGB2BGR)            
            images = np.hstack((frame_contour, depth_color_image))
            
            # show images in opencv window
            show_image(images)
            
            
            
            
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                file.close()
                break
            
            
            










# load example ROS bag  
def default_bag_input():
    #Get the current working directory
    cwd = os.getcwd()
    cwd = cwd.replace(os.sep,"/")
    inputbag = cwd + "/bags/pre_session_cali_1270_720_edge.bag"
    
    return inputbag


# Rescale image to fit window size
def rescale_image_by_ratio(image,rescale_percentage=0.5):
    resized_h = int(image.shape[0]*rescale_percentage)
    resized_w = int(image.shape[1]*rescale_percentage)
    resize_dim = (resized_w, resized_h)
    resized_image = cv.resize(image, resize_dim)
    
    return resized_image
        
 
### Show images
def show_image(image,rescale_percentage=0.5):
    if image.shape[1] > 1280 or image.shape[0] > 720:
        image = rescale_image_by_ratio(image,rescale_percentage)
        
    cv.namedWindow("Stream", cv.WINDOW_AUTOSIZE)
    cv.imshow("Stream", image)
            

### IN PROGRESS
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


### IN PROGRESS
# Post processing filtering
def post_processing(depth_frame):
    spat_filter = rs.spatial_filter()       # Spatial    - edge-preserving spatial smoothing
    temp_filter = rs.temporal_filter()      # Temporal   - reduces temporal noise
    hole_fill = rs.hole_filling_filter()    # Hole-filling filter
    
    filtered = spat_filter.process(depth_frame)
    filtered = temp_filter.process(filtered)
    depth_frame_filtered = hole_fill.process(filtered)


# RGB histogram plotting
def rgb_histogram(channels):
    
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


### IN PROGRESS
# rgb masking
def rgb_masking(color_image):
    # rgb channels
    R, G, B = cv.split(color_image)

    # convert to floating point
    fR = np.asarray(R, dtype=float)
    fG = np.asarray(G, dtype=float)
    fB = np.asarray(B, dtype=float)
    
    #normalised 
    my_eps = 0.001 # Avoids divide by zero
    denominator = (fR+fG+fB+my_eps)
    nR = fR/denominator
    nG = fG/denominator
    nB = fB/denominator
    # plot histogram
    # rgb_histogram([nR, nG, nB])
   

    # binary images with threshold
    mask_r = nR > 0.39 # 0.4
    mask_g = nG > 0.41 # 0.4
    mask_b = nB > 0.35 # 0.37

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
    
   
    return mask_r_int, mask_g_int, mask_b_int


# Morphological transformation
def morpho_trans(frame):

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


### IN PROGRESS
# Tracking box
def drawing_tracking_box(image):
    pass


## IN PROGRESS
# accuracy test
def accuracy_tests():
    pass


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
    
    

