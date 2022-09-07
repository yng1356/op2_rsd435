# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import animation
import math


from sklearn import mixture, cluster


class ColorDetect:
    
    # 2d scatter plots - RG, RB, GB slices
    def rgb_histogram(self, img):
        
        colours = ("red", "green", "blue")
        channel_ids = (0, 1, 2)
        
        plt.figure(figsize=(8, 6), dpi=80)
        plt.xlim([0, 256])
        for (channel_id, c) in zip(channel_ids, colours):
            histogram, bin_edges = np.histogram(img[:, :, channel_id], bins=256, range=(0,255))
            plt.plot(histogram, color=c, label=c)
        
        plt.xlabel("colour value", fontsize=18)
        plt.ylabel("pixel count", fontsize=18)
        plt.legend(loc='upper left', fontsize=18)
        plt.show()
        
    
    # manual rgb masking 
    def rgb_masking(self, img):
        img_mask = img.copy()
        
        rgb_list = ['Reds','Greens','Blues']
        fig, ax = plt.subplots(1, 3, figsize=(15,5), sharey = True)
        for i in range(3):
            ax[i].imshow(img_mask[:,:,i], cmap = rgb_list[i])
            ax[i].set_title(rgb_list[i], fontsize = 15)
        
        # RGB masks
        mask_r = (img_mask[:,:,0] > 150) & (img_mask[:,:,1] < 120) & (img_mask[:,:,2] < 140)
        mask_g = (img_mask[:,:,1] > 100) & (img_mask[:,:,0] < 75) & (img_mask[:,:,2] < 130)
        mask_b = (img_mask[:,:,2] > 100) & (img_mask[:,:,0] < 60) & (img_mask[:,:,1] < 120)
        
        img_mask[:, :, 0] = img_mask[:, :, 0]*mask_r
        img_mask[:, :, 1] = img_mask[:, :, 1]*mask_g
        img_mask[:, :, 2] = img_mask[:, :, 2]*mask_b
        
        plt.figure(figsize=(8, 6), dpi=80)
        plt.imshow(img_mask)
        
        return img_mask
    
    
    # manual rgb masking 
    def hsv_masking(self, img):
        img_mask = img.copy()
        img_hsv = img.copy()
        hsv_mask = img.copy()
        hsv_mask = cv.cvtColor(hsv_mask, cv.COLOR_BGR2HSV_FULL)
        
        hsv_list = ['Hue', 'Saturation', 'Value']
        cmap_list = ['hsv', 'Greys', 'gray']
        fig, ax = plt.subplots(1, 3, figsize=(15,7), sharey = True)
        for i in range(3):
            ax[i].imshow(hsv_mask[:,:,i], cmap = cmap_list[i])
            ax[i].set_title(hsv_list[i], fontsize = 20)
            ax[i].axis('off')
        fig.tight_layout()
        plt.show()
        
        # HSV masks
        mask_r = (hsv_mask[:,:,0] > 160) & (hsv_mask[:,:,1] > 100)
        mask_g = (hsv_mask[:,:,0] > 30) & (hsv_mask[:,:,0] < 75) & (hsv_mask[:,:,1] > 100)
        mask_b = (hsv_mask[:,:,0] < 30) & (hsv_mask[:,:,1] > 100)
        
        img_hsv[:, :, 0] = img_hsv[:, :, 0]*mask_r
        img_hsv[:, :, 1] = img_hsv[:, :, 1]*mask_g
        img_hsv[:, :, 2] = img_hsv[:, :, 2]*mask_b
        
        plt.figure(figsize=(8, 6), dpi=80)
        plt.imshow(img_hsv)
        
        for i in np.arange(0, 360):
            for j in np.arange(0,512):
                if ((hsv_mask[i,j,0]> 160 or (hsv_mask[i,j,0]<75))and hsv_mask[i,j,1]>100):
                    pass
                else:
                    img_mask[i,j,:] = [0,0,0]
        
        plt.figure(figsize=(8, 6), dpi=80)
        plt.imshow(img_mask)
        
        
        return img_mask, mask_r, mask_g, mask_b
        
    
    # 2d rgb scatter plot   
    def scatter_2d(self, img, stepsize=2000):
        
        # plot the filtered pixels
        flatten_dataset = img.reshape((-1, 3))
        maxlength = np.shape(flatten_dataset)[0]
        
        
        img_hsv = img.copy()
        img_hsv = cv.cvtColor(img_hsv, cv.COLOR_BGR2HSV_FULL)
        h, s, v = cv.split(img_hsv)
        h = h.flatten()
        s = s.flatten()
        v = v.flatten()
        
        
        # fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(24,8))
        
        for i in np.arange(0, maxlength, stepsize):
            # if ((h[i]> 165 or (h[i]>30 and h[i]<75) or h[i]<30) and s[i]>100):
            p = flatten_dataset[i,:]
            color = '#%02x%02x%02x'%(p[0], p[1], p[2])

            # ax1.scatter(p[0], p[1],c=color)
            # ax2.scatter(p[0], p[2],c=color)
            # ax3.scatter(p[1], p[2],c=color)
            
            # ax1.scatter(h[i], s[i],c=color)
            # ax2.scatter(h[i], v[i],c=color)
            # ax3.scatter(s[i], v[i],c=color)
        
        # ax1.set_title('RG', size=18)
        # ax2.set_title('RB', size=18)
        # ax3.set_title('GB', size=18)
        
        # ax1.set_xlabel('Red', fontsize=18)
        # ax1.set_ylabel('Green', fontsize=18)
        # ax2.set_xlabel('Red', fontsize=18)
        # ax2.set_ylabel('Blue', fontsize=18)
        # ax3.set_xlabel('Green', fontsize=18)
        # ax3.set_ylabel('Blue', fontsize=18)
        
        # ax1.set_title('HS', size=18)
        # ax2.set_title('HV', size=18)
        # ax3.set_title('SV', size=18)
        
        # ax1.set_xlabel('Hue', fontsize=18)
        # ax1.set_ylabel('Saturation', fontsize=18)
        # ax2.set_xlabel('Hue', fontsize=18)
        # ax2.set_ylabel('Value', fontsize=18)
        # ax3.set_xlabel('Saturation', fontsize=18)
        # ax3.set_ylabel('Value', fontsize=18)

        # plt.setp((ax1, ax2, ax3), xticks=np.linspace(0,255,18), yticks=np.linspace(0,255,18))    
        # plt.show()

    
    
    # 3d rgb scatter plot
    def scatter_3d(self, img, stepsize=2000):
        flatten_dataset = img.reshape((-1, 3))
        maxlength = np.shape(flatten_dataset)[0]
    
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.view_init(elev=10, azim=240) # 10, 240
        plt.draw()
        
        for i in np.arange(0, maxlength, stepsize):
            p = flatten_dataset[i,:]
            color = '#%02x%02x%02x'%(p[0], p[1], p[2])
            
            ax.scatter(p[0], p[1], p[2],c=color)
            
        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")
        # fig.suptitle('RGB value scatter plot', fontsize=20)
        
        plt.show()
        
  
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
        rot_animation.save('3d_scatter_5.gif')   


    
    def get_mask(self, img, roi, single_cluster, points_idx):
        
        H, W = np.shape(img)[0], np.shape(img)[1]
        mask= np.zeros((H,W,3))
        
        for p in range(np.shape(single_cluster)[0]):
            colour_values = single_cluster[p]
            colour2pixel = self.colour_to_coordinates(roi, single_cluster[p], points_idx[p])
            mask[colour2pixel[0],colour2pixel[1],:] = [colour_values[0],colour_values[1],colour_values[2]]
        
        # mask = mask.astype(np.uint8)
        # plt.matshow(mask)  
        
        return mask

    
    
    # convert rgb/hsv values back to its pixel coordinate
    def colour_to_coordinates(self, roi, rgb_point, idx):

        [x, y, w, h] = roi

        # find index of target point
        row = int(np.floor(idx/w))
        col = idx%w

        origianl_x = col+x
        origianl_y = row+y
        
        pixel_coordinate=[origianl_y,origianl_x]
    
        return pixel_coordinate


    # rgb clustering - kmeans
    def rgb_kmeans(self, flatten_dataset, k=4):
        
        # cv.kmeans
        flatten_dataset = np.float32(flatten_dataset)
        
        
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
    
        _, labels, centers = cv.kmeans(data=flatten_dataset,
                                            K=k,
                                            bestLabels=None,
                                            criteria=criteria,
                                            attempts=10,
                                            flags=cv.KMEANS_RANDOM_CENTERS
                                            )
        
        
        
        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(18,6))
        
        
        ax1.scatter(flatten_dataset[:, 0], flatten_dataset[:, 1], c=labels, s=40)
        ax2.scatter(flatten_dataset[:, 0], flatten_dataset[:, 2], c=labels, s=40)
        ax3.scatter(flatten_dataset[:, 1], flatten_dataset[:, 2], c=labels, s=40);
        

        plt.setp((ax1, ax2, ax3), xticks=np.linspace(0,255,18), yticks=np.linspace(0,255,18))    
        plt.show()
        
        
        return 
    
    
    # Gaussian mixture clustering 
    def rgb_GMM(self,flatten_dataset, n_components=5):
        # plt.imshow(img)
        
        gmm = mixture.GaussianMixture(n_components,covariance_type='full').fit(flatten_dataset)
        labels = gmm.predict(flatten_dataset)
        
        
        # cmap='viridis'
        
        fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, sharex=True, figsize=(24,8))
        
       
        ax4.scatter(flatten_dataset[:,0], flatten_dataset[:,1],c=labels)
        ax5.scatter(flatten_dataset[:,0], flatten_dataset[:,2],c=labels)
        ax6.scatter(flatten_dataset[:,1], flatten_dataset[:,2],c=labels)
        
       
        ax4.set_title('HS', size=18)
        ax5.set_title('HV', size=18)
        ax6.set_title('SV', size=18)
        
        ax4.set_xlabel('Hue', fontsize=18)
        ax4.set_ylabel('Saturation', fontsize=18)
        ax5.set_xlabel('Hue', fontsize=18)
        ax5.set_ylabel('Value', fontsize=18)
        ax6.set_xlabel('Saturation', fontsize=18)
        ax6.set_ylabel('Value', fontsize=18)

        plt.setp((ax4, ax5, ax6), xticks=np.linspace(0,255,18), yticks=np.linspace(0,255,18))    
        plt.show()
        
        # n_components = np.arange(3, 21, 1)
        # models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(flatten_dataset)
        #           for n in n_components]
        # fig = plt.figure()
        # plt.plot(n_components, [m.bic(flatten_dataset) for m in models], label='BIC')
        # plt.plot(n_components, [m.aic(flatten_dataset) for m in models], label='AIC')
        # plt.legend(loc='best')
        # plt.xlabel('n_components')
        # plt.show()
        
        
    def rgb_DBSCAN(self,flatten_dataset, points_idx, n_components=3):        
        
        
        dbscan = cluster.DBSCAN(eps=5, min_samples=10)
        clusters = dbscan.fit_predict(flatten_dataset)
        labels = dbscan.labels_
        
        fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, sharex=True, figsize=(24,8))
        
        ax4.scatter(flatten_dataset[:,0], flatten_dataset[:,1],c=clusters,alpha=0.3)
        ax5.scatter(flatten_dataset[:,0], flatten_dataset[:,2],c=clusters,alpha=0.3)
        ax6.scatter(flatten_dataset[:,1], flatten_dataset[:,2],c=clusters,alpha=0.3)
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # print(n_clusters)
        # print(n_noise)
        
        # sort the clusters in order of amounts and get the top three biggest clusters
        counts = np.bincount(labels[labels>=0])
        # print(counts)
        top_labels = np.argsort(-counts)[:3]

        
        cluster_0 = flatten_dataset[labels == top_labels[0]]
        cluster_1 = flatten_dataset[labels == top_labels[1]]
        cluster_2 = flatten_dataset[labels == top_labels[2]]
        
        # get centroid for each cluster
        centroid_0 = cluster_0.mean(axis=0)
        centroid_1 = cluster_1.mean(axis=0)
        centroid_2 = cluster_2.mean(axis=0)
        centroids = np.array([centroid_0,centroid_1,centroid_2])
        
        # reassign labels to each colour clusters
        centroid_h = centroids[:,0]
        sorted_labels = top_labels[np.argsort(centroid_h)]

        
        cluster_r = flatten_dataset[labels == sorted_labels[2]]
        cluster_g = flatten_dataset[labels == sorted_labels[1]]
        cluster_b = flatten_dataset[labels == sorted_labels[0]]
        
        centroids = centroids[np.argsort(centroid_h)[::-1]]
        # print(centroids)

        colours = ['r','g','b']
        for i in range(3):
            ax4.scatter(centroids[i,0], centroids[i,1], c=colours[i], s=300, marker='x',linewidths=3)
            ax5.scatter(centroids[i,0], centroids[i,2], c=colours[i], s=300, marker='x',linewidths=3)
            ax6.scatter(centroids[i,1], centroids[i,2], c=colours[i], s=300, marker='x',linewidths=3)
    
        
        
        plt.setp((ax4, ax5, ax6), xticks=np.linspace(0,255,18), yticks=np.linspace(0,255,18))    
        plt.show()
        
        points_idx = np.array(points_idx)
        
        idx_r = points_idx[labels == sorted_labels[2]]
        idx_g = points_idx[labels == sorted_labels[1]]
        idx_b = points_idx[labels == sorted_labels[0]]

        
        return cluster_r, cluster_g, cluster_b, idx_r, idx_g, idx_b, centroids
       
        
        
        
        # print("Estimated number of clusters: %d" % n_clusters_)
        # print("Estimated number of noise points: %d" % n_noise_)
        # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
        # print(
        #     "Adjusted Mutual Information: %0.3f"
        #     % metrics.adjusted_mutual_info_score(labels_true, labels)
        # )
        # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
        
        

        
        
        
        