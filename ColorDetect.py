# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import animation
import math


from sklearn import mixture, cluster


class ColorDetect:

# NOT USED ANYMORE
    # RGB histogram plotting - biased
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
    
# NOT USED ANYMORE    
    # rgb masking - biased
    def rgb_filtering(color_image):
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
        mask_r = nR > 0.45 # 0.39
        mask_g = nG > 0.4 # 0.41
        mask_b = nB > 0.45 # 0.37
    
        # plt.figure(figsize=(12,3))
        # plt.subplot(1,3,1)
        # plt.imshow(255*mask_r, interpolation=None) #'none'
        # plt.subplot(1,3,2)
        # plt.imshow(255*mask_g, interpolation=None)
        # plt.subplot(1,3,3)
        # plt.imshow(255*mask_b, interpolation=None)   
                
        
        # convert to uint8 format
        mask_r_int = mask_r.astype(np.uint8)*255
        mask_g_int = mask_g.astype(np.uint8)*255
        mask_b_int = mask_b.astype(np.uint8)*255
        
       
        return mask_r_int, mask_g_int, mask_b_int
    
    
    # 2d scatter plots - RG, RB, GB slices
    def scatter_2d(self, flatten_dataset, stepsize=2000):
        maxlength = np.shape(flatten_dataset)[0]
        
        filtered_points = []
        points_idx = []
    
        
        # fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(24,8))
        # fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, sharex=True, figsize=(24,8))
        
        for i in np.arange(0,maxlength,stepsize): 
            p = flatten_dataset[i,:]
    
            color = '#%02x%02x%02x'%(p[0], p[1], p[2])
            
            if (p[0]>=150 and p[1]<120 and p[2]<130):
                filtered_points.append(p)
                points_idx.append(i)
                
                # ax1.scatter(p[0], p[1],c=color)
                # ax2.scatter(p[0], p[2],c=color)
                # ax3.scatter(p[1], p[2],c=color)
                
            
            elif (p[1]>=110 and p[0]<75 and p[2]<130):
                filtered_points.append(p)
                points_idx.append(i)
                
                # ax1.scatter(p[0], p[1],c=color)
                # ax2.scatter(p[0], p[2],c=color)
                # ax3.scatter(p[1], p[2],c=color)
                
            elif (p[2]>=100 and p[0]<60 and p[1]<120):
                filtered_points.append(p)
                points_idx.append(i)
                
                # ax1.scatter(p[0], p[1],c=color)
                # ax2.scatter(p[0], p[2],c=color)
                # ax3.scatter(p[1], p[2],c=color)
            else:
                pass
        
            # ax4.scatter(p[0], p[1],c=color)
            # ax5.scatter(p[0], p[2],c=color)
            # ax6.scatter(p[1], p[2],c=color)
            
        # ax1.set_title('RG', size=18)
        # ax2.set_title('RB', size=18)
        # ax3.set_title('GB', size=18)
        
        # ax1.set_xlabel('Red', fontsize=18)
        # ax1.set_ylabel('Green', fontsize=18)
        # ax2.set_xlabel('Red', fontsize=18)
        # ax2.set_ylabel('Blue', fontsize=18)
        # ax3.set_xlabel('Green', fontsize=18)
        # ax3.set_ylabel('Blue', fontsize=18)
        
        
        # plt.setp((ax1, ax2, ax3), xticks=np.linspace(0,255,18), yticks=np.linspace(0,255,18))    
        plt.show()
            
        
        return filtered_points, points_idx
    

# NOT DONE YET   
    # 3d scatter plot
    def scatter_3d(flatten_dataset, stepsize=2000):
        maxlength = np.shape(flatten_dataset)[0]
    
        filtered_points = []
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.view_init(elev=10, azim=240) # 10, 240
        plt.draw()
        for p in flatten_dataset[0:maxlength:stepsize,:]:
            if (p[0]>=150 and p[1]<90 and p[2]<100) or (p[1]>=110 and p[0]<70 and p[2]<120) or (p[2]>=140 and p[0]<70 and p[1]<120):
                filtered_points.append(p)
                ax.scatter(p[0], p[1], p[2], c='#%02x%02x%02x'%(p[0], p[1], p[2]))
        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")
        fig.suptitle('RGB value scatter plot', fontsize=20)
        
        filtered_points = np.array(filtered_points)
        
        return filtered_points
  
    
    # rgb 3d scatter plotting
    def rgb_scatter_plotting(self, img, stepsize=2000):
        
        pixel_values = img.reshape((-1, 3))
        # print(np.shape(pixel_values))
        
        # filtered_points = scatter_3d(pixel_values, stepsize=20000)
        filtered_points, points_idx = self.scatter_2d(pixel_values, stepsize)
        # rotation_animation()
        
        return filtered_points, points_idx
    
  
    # rotation animation for 3d plot 
    def rotation_animation():
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        # for angle in range(0, 360):
        #     ax.view_init(elev=10, azim=angle) # 10, 240
        #     plt.draw()
        #     plt.pause(.001)
    
        def rotate(angle):
            ax.view_init(elev=10, azim=-angle)
        rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
        rot_animation.save('3d_scatter_4.gif')
    
    
    def seperate_rgb_masks(self, H, W, resize_img, roi, single_cluster, idx):
        mask= np.zeros((H,W,3))
        
        for p in range(np.shape(single_cluster)[0]):
            rgb2pixel = self.rgb_to_coordinates(resize_img, roi, single_cluster[p], idx[p])
            rgb_values = single_cluster[p]
            
            mask[rgb2pixel[0],rgb2pixel[1]] = [rgb_values[0],rgb_values[1],rgb_values[2]]
        
        mask = mask.astype(np.uint8)
        
        # plt.matshow(mask)  
        
        return mask
 
    
    # convert rgb values back to pixels
    def rgb_to_coordinates(self, img, roi, rgb_point, idx):
        rows = np.shape(img)[0]
        cols = np.shape(img)[1]
        
        [x, y, w, h] = roi
        
        # find index of target point
        row = int(np.floor(idx/cols))
        col = idx%cols
        
        origianl_x = col+x
        origianl_y = row+y
        
        pixel_coordinate=[origianl_y,origianl_x]
    
        return pixel_coordinate

    
    # rgb clustering - kmeans
    def rgb_kmeans(self, filtered_points, points_idx, k=4):
        
        # cv.kmeans
        filtered = np.float32(filtered_points)
        
        
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
    
        _, labels, centers = cv.kmeans(data=filtered,
                                            K=k,
                                            bestLabels=None,
                                            criteria=criteria,
                                            attempts=10,
                                            flags=cv.KMEANS_RANDOM_CENTERS
                                            )
        
        
        
        # fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(18,6))
        
        sorted_d = []
        
        for i in range(k):
            centre = centers[i]
            d = np.sqrt((centre[0])**2+(centre[1])**2+(centre[2])**2)
            
            # print("label", i ,"center",centre,"d", d)
            
            sorted_d.append(d)
            
        sorted_i = np.argsort(sorted_d)
        # print(sorted_i)
        
        labels = np.array(labels).ravel()
        
        cluster_r = filtered[labels==sorted_i[2]]
        cluster_g = filtered[labels==sorted_i[0]]
        cluster_b = filtered[labels==sorted_i[1]]
        
        
        points_idx = np.array(points_idx)
        
        idx_r = points_idx[np.array(np.where(labels==sorted_i[2])).flatten()]
        idx_g = points_idx[np.array(np.where(labels==sorted_i[0])).flatten()]
        idx_b = points_idx[np.array(np.where(labels==sorted_i[1])).flatten()]
        
        
        
        # for i in sorted_i:
              
        #     cluster_i = filtered[labels==sorted_i[i]]
           
        #     ax1.scatter(cluster_i[:,0],cluster_i[:,1])
        #     ax2.scatter(cluster_i[:,0],cluster_i[:,2])
        #     ax3.scatter(cluster_i[:,1],cluster_i[:,2])
            
        # ax1.scatter(centers[:,0],centers[:,1],s = 80,c = 'y', marker = 's')
        # ax2.scatter(centers[:,0],centers[:,2],s = 80,c = 'y', marker = 's')
        # ax3.scatter(centers[:,1],centers[:,2],s = 80,c = 'y', marker = 's')
        
        # ax1.set_title('RG', size=18)
        # ax2.set_title('RB', size=18)
        # ax3.set_title('GB', size=18)
        
        # ax1.set_xlabel('Red', fontsize=18)
        # ax1.set_ylabel('Green', fontsize=18)
        # ax2.set_xlabel('Red', fontsize=18)
        # ax2.set_ylabel('Blue', fontsize=18)
        # ax3.set_xlabel('Green', fontsize=18)
        # ax3.set_ylabel('Blue', fontsize=18)
        
        # plt.setp((ax1, ax2, ax3), xticks=np.linspace(0,255,18), yticks=np.linspace(0,255,18)) #4 
        # fig1.suptitle('RGB value Kmeans clustering', fontsize=20)
        # fig1.tight_layout()
        
        # plt.show()
        
        
        
        return cluster_r, cluster_g, cluster_b, idx_r, idx_g, idx_b
    
    
# NOT DONE YET
    # rgb clustering - Gaussian Mixture     
    def rgb_GMM(img, n_components=10, p_thershold=0.95):
        
        H = np.shape(img)[0]
        W = np.shape(img)[1]
        
        lb_H = int(np.round(0.2*H)) # 72
        ub_H = int(np.round(0.8*H)) # 648
        lb_W = int(np.round(0.2*W)) # 128
        ub_W = int(np.round(0.8*W)) # 1152
        
        pv = img[lb_H:ub_H, lb_W:ub_W,:]
        
        pv = pv.reshape((-1, 3))
        print(np.shape(pv))
        
        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(18,6))
        
        
        # define the model
        gmm = mixture.GaussianMixture(n_components=10, covariance_type='full', random_state=0)
        # fit the model
        gmm.fit(pv)
    
        global prob
        prob = gmm.predict_proba(pv)
    
    
        # assign a cluster to each example
        yhat = gmm.predict(pv)
        # retrieve unique clusters
        clusters = np.unique(yhat)
    
        # create scatter plot for samples from each cluster
        for cluster in clusters:
            # get row indexes for samples with this cluster
            row_ix = np.where(yhat == cluster)
            
            p = np.where(prob[:,cluster] > p_thershold)
            
            row_ix = np.intersect1d(row_ix,p)
            
            # create scatter of these samples
            ax1.scatter(pv[row_ix, 0], pv[row_ix, 1], alpha=0.1)
            ax2.scatter(pv[row_ix, 0], pv[row_ix, 2], alpha=0.1)
            ax3.scatter(pv[row_ix, 1], pv[row_ix, 2], alpha=0.1)
            
        # ax1.scatter(clusters[:,0],clusters[:,1],s = 80,c = 'y', marker = 's')
        # ax2.scatter(clusters[:,0],clusters[:,2],s = 80,c = 'y', marker = 's')
        # ax3.scatter(clusters[:,1],clusters[:,2],s = 80,c = 'y', marker = 's')
        
        
            
        # show the plot
        plt.show()
    


    ## HSV try-on
    # HSV scatter
    def hsv_scatter_potting(self, img,stepsize=2000):
        pixel_values = img.reshape((-1, 3))
        maxlength = np.shape(pixel_values)[0]
        
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
        
        h, s, v = cv.split(img_hsv)
        
        h = h.flatten()
        s= s.flatten()
        v= v.flatten()
        
        filtered_points = []
        
        
        fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, sharex=True, figsize=(24,8))
        
        for i in np.arange(0,maxlength,stepsize): 
            
            if (h[i]< 100 or h[i]>160) and s[i]>80:
                
                p = pixel_values[i,:]
                filtered_points.append([h[i],s[i],v[i]])
            
                color = '#%02x%02x%02x'%(p[0], p[1], p[2])
            
                ax4.scatter(h[i], s[i],c=color)
                ax5.scatter(h[i], v[i],c=color)
                ax6.scatter(s[i], v[i],c=color)
            
       
        ax4.set_title('HS', size=18)
        ax5.set_title('HV', size=18)
        ax6.set_title('SV', size=18)
        
        ax4.set_xlabel('Hue', fontsize=18)
        ax4.set_ylabel('Saturation', fontsize=18)
        ax5.set_xlabel('Hue', fontsize=18)
        ax5.set_ylabel('Value', fontsize=18)
        ax6.set_xlabel('Saturation', fontsize=18)
        ax6.set_ylabel('Value', fontsize=18)
        
       
        # plt.setp((ax1, ax2, ax3), xticks=np.linspace(0,255,18), yticks=np.linspace(0,255,18))    
        
        plt.setp((ax4, ax5, ax6), xticks=np.linspace(0,255,18), yticks=np.linspace(0,255,18)) #4 
        fig2.suptitle('HSV scatter plot', fontsize=20)
        fig2.tight_layout()
        
        plt.show()
        
        return filtered_points
            
      
    # rgb clustering - kmeans
    def hsv_kmeans(self, filtered_points, k=4):
        
        # cv.kmeans
        filtered = np.float32(filtered_points)
        print(np.shape(filtered))
        
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
    
        _, labels, centers = cv.kmeans(data=filtered,
                                            K=k,
                                            bestLabels=None,
                                            criteria=criteria,
                                            attempts=10,
                                            flags=cv.KMEANS_RANDOM_CENTERS
                                            )
        
        
        
        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(18,6))

        for i in labels:   
            cluster_i = filtered[labels==i]
            ax1.scatter(cluster_i[:,0],cluster_i[:,1])
            ax2.scatter(cluster_i[:,0],cluster_i[:,2])
            ax3.scatter(cluster_i[:,1],cluster_i[:,2])
            
        ax1.scatter(centers[:,0],centers[:,1],s = 80,c = 'k', marker = 's')
        ax2.scatter(centers[:,0],centers[:,2],s = 80,c = 'k', marker = 's')
        ax3.scatter(centers[:,1],centers[:,2],s = 80,c = 'k', marker = 's')
        
        ax1.set_title('HS', size=18)
        ax2.set_title('HV', size=18)
        ax3.set_title('SV', size=18)
        
        ax1.set_xlabel('Hue', fontsize=18)
        ax1.set_ylabel('Saturation', fontsize=18)
        ax2.set_xlabel('Hue', fontsize=18)
        ax2.set_ylabel('Value', fontsize=18)
        ax3.set_xlabel('Saturation', fontsize=18)
        ax3.set_ylabel('Value', fontsize=18)
        
        plt.setp((ax1, ax2, ax3), xticks=np.linspace(0,255,18), yticks=np.linspace(0,255,18)) #4 
        fig1.suptitle('HSV value Kmeans clustering', fontsize=20)
        fig1.tight_layout()
        
        plt.show()
        
        
        