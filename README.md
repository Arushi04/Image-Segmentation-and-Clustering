# Image-Segmentation-and-Clustering

### Description:

This project focuses on finding color details of handbags and their logos. The project is done in 3 steps:
1. Handbag and logo detection using Google vision APIs   
2. Segmenting the detected region into superpixels (cluster based on color similarity and physical proximity). And then averaging the color in each superpixel.
3. Clustering the superpixels by using K-means clustering to find clusters by color similarity and finding the average cluster colors and the proportion of pixels assigned to them.   
Below features have been calculated for all handbags:
1.	Logo size : size of the logo
2.	Logo contrast : contrast ratio of the most dominant color of logo to handbag
3.	Logo conspicuousness : contrast ratio * relative size of the logo
4.	Total number of colors in the handbag
5.	Color Entropy of Handbag : to calculate the distribution of the colors


### Installation Requirements:
1. google-cloud-vision==2.0.0
2. google-auth-oauthlib==0.4.1
3. scikit-image==0.16.2
4. webcolors==1.9.1
5. pandas==1.1.0
6. numpy==1.19.1
7. sklearn==0.23.1
8. PIL==7.2.0
9. cv2==3.4.2


### Steps to run:    
python main.py \
--dir_path <directory path> \
--bag_segments <no of segments for bag segmentation> \
--bag_cluster_size <cluster size for bag> \
--logo_segments <no of segments for logo segmentation> \
--logo_cluster_size <cluster size for logo>
  




