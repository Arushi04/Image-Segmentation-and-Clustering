import webcolors
import numpy as np
import cv2


def closest_color(requested_color):
    '''
     This functions takes a color in rgb and returns the color that closely resembles to that color
     Input: Color
     Output: color closest to the minimum color
    '''

    min_colors = {}
    for key, name in webcolors.css21_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name

    if min_colors[min(min_colors.keys())] == 'gray':
        for key, name in webcolors.css3_hex_to_names.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_colors[(rd + gd + bd)] = name

    return min_colors[min(min_colors.keys())]


def get_color_name(requested_color):
    '''
     This functions takes a color in rgb and returns the name of the color.
     Input: Color
     Output: Name of the color
    '''

    try:
        actual_name = webcolors.rgb_to_name(requested_color)
        closest_name = None
    except ValueError:
        closest_name = closest_color(requested_color)
        actual_name = None

    return actual_name, closest_name


def centroid_histogram(clt):
    '''
     This functions takes the labels from the kmeans model and creates a histogram
     Input: fitted model
     Output: creates a histogram of all the colors based on labels
    '''
    # grab the number of different clusters based on the number of pixels assigned to each cluster

    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def get_most_dominant_color(bar):
    '''
     This functions takes all the colors and returns the most dominant color, ratio and rgb_color
     Input: all the colors
     Output: Most dominant color with its ratio and rgb value
    '''

    dominant_color = "black"
    max_ratio = 0

    for colors in bar:
        if colors[1] > max_ratio:
            max_ratio = colors[1]
            dominant_color = colors[0]
            rgb_color = colors[2]

    return dominant_color, max_ratio, rgb_color


def get_sorted_colors(all_colors):
    '''
     This functions takes all the colors which can be duplicate and returns the list of unique colors in
     decending order based on their total ratio
     Input: all the colors
     Output: list of unique colors in decending order
    '''

    color_pixels = {}

    for color in all_colors:
        if color[0] not in color_pixels:
            color_pixels[color[0]] = color[1]
        else:
            color_pixels[color[0]] += color[1]

    sorted_colors_pixels = sorted(color_pixels.items(), key=lambda x: x[1], reverse=True)

    return sorted_colors_pixels


def get_colors(hist, centroids):
    '''
     This functions takes the histogram and labels and returns dominant color, it's ratio, rgb value and all colors
     Input: histogram and labels
     Output: dominant_color, ratio, rgb_color, all_colors
    '''

    # initialize the bar chart representing the relative frequency of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    all_colors = []

    # loop over the percentage and color of each cluster
    for (percent, color) in zip(hist, centroids):
        requested_color = color.astype(int)

        actual_name, closest_name = get_color_name(requested_color)

        if actual_name is None:
            all_colors.append([closest_name, percent, requested_color])
        else:
            all_colors.append([actual_name, percent, requested_color])

        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    dominant_color, ratio, rgb_color = get_most_dominant_color(all_colors)

    return bar, dominant_color, ratio, rgb_color, all_colors


def calculate_luminace(color_code):
    '''
    This function calculates the luminance of color
    Input: a color code in rgb
    Output: luminance
    '''

    index = float(color_code) / 255

    if index < 0.03928:
        return index / 12.92
    else:
        return ((index + 0.055) / 1.055) ** 2.4


def calculate_relative_luminance(rgb):
    '''
    This function calculates the relative luminance of the color using the 3 channels of R, G, B
    Input: a color in rgb
    Output: relative luminance
    '''

    return 0.2126 * calculate_luminace(rgb[0]) + 0.7152 * calculate_luminace(rgb[1]) + \
           0.0722 * calculate_luminace(rgb[2])


def is_white_pixel(x):
    x = x.tolist()
    white_range = [250, 251, 252, 253, 254, 255]
    if x[0] in white_range and x[1] in white_range and x[2] in white_range:
    #if x[0]==x[1]==x[2]==255:
        return False
    return True
