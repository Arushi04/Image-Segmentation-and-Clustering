import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from sklearn.cluster import KMeans

from utils import centroid_histogram, get_colors, is_white_pixel


def segment_image(cropped_img, numSegments):
    '''
    Segements the cropped image as input and no of segments and convert it into superpixels
    Input: Cropped Image, Number of segments
    Output: Segmented image and image with average of colors in superpixels
    '''

    converted_img = img_as_float(cropped_img[:, :, ::-1])  # convert it to a floating point data type
    segments = slic(converted_img, n_segments=numSegments, compactness=10, sigma=5, convert2lab=True)

    # Average the color in each superpixel.
    out_img = mean_image(cropped_img, segments)

    return out_img


def segment_and_cluster(num_segments, cropped_img, cluster_size):
    '''
    This function segments the image, cluster the pixels using kmeans clustering and return the complete details of colors
    Input: number of segments to be done, cropped image, cluster size
    Output: bar, dominant_color, ratio, rgb_color, all_colors
    '''

    # Segmentation of the image into superpixels and taking average of superpixels
    out_img = segment_image(cropped_img, num_segments)

    # Removing the white background before clustering
    reshaped_img = out_img.reshape((-1, 3))
    mask = np.apply_along_axis(is_white_pixel, 1, reshaped_img)
    white_removed = reshaped_img[mask]

    # cluster the pixel intensities
    clt = KMeans(n_clusters=cluster_size, random_state=42)
    clt.fit(white_removed)

    hist = centroid_histogram(clt)
    bar, dominant_color, ratio, rgb_color, all_colors = get_colors(hist, clt.cluster_centers_)

    return bar, dominant_color, ratio, rgb_color, all_colors


def mean_image(image, segments):
    '''
    This function takes segmented image and no of segments and returns the image with average of superpixels
    Input: Segmented Image, No of segments
    Output: Image with average of colors in superpixels
    '''

    reshaped_image = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
    segment_1d = np.reshape(segments, -1)
    unique_segment = np.unique(segment_1d)
    img_shape = np.zeros(reshaped_image.shape)

    for i in unique_segment:
        loc = np.where(segment_1d == i)[0]
        img_mean = np.mean(reshaped_image[loc, :], axis=0)
        img_shape[loc, :] = img_mean

    out_img = np.reshape(img_shape, [image.shape[0], image.shape[1], image.shape[2]]).astype('uint8')

    return out_img