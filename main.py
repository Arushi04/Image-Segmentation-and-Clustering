import argparse
import pandas as pd
import os
import math
import numpy as np
from PIL import Image as Img

from vision import crop_image, get_logo
from utils import get_sorted_colors, calculate_relative_luminance
from segment import segment_and_cluster


# Connecting to Google Cloud
from google.cloud import vision
from google.oauth2 import service_account
creds = service_account.Credentials.from_service_account_file('My First Project-c1dd0c0c9fd5.json')
client = vision.ImageAnnotatorClient(credentials=creds)

def process_images(img_path):
    '''
    This is the main function that takes the image path as an input, processes it and returns the final result with 5 color features
    Input : image path
    Output : logo_size, contrast_ratio, logo_conspicuousness, total_colors, entropy
    '''

    if not os.path.isfile(img_path):
        return pd.Series([None] * 5)

    # Get the cropped bag
    img = Img.open(img_path)
    cropped_bag = crop_image(img, img_path)
    height, width = cropped_bag.size
    bag_size = height * width
    # Img.fromarray(np.array(cropped_bag)).show()

    # Get the cropped logo
    cropped_img_path = 'cropped.jpg'
    cropped_img = Img.open(cropped_img_path)
    cropped_logo = get_logo(cropped_img, cropped_img_path)

    # converting cropped bag img from PIL to numpy array
    cropped_bag = np.array(cropped_bag)

    # Segmentation of the image into superpixels, taking average of superpixels and then doing clustering
    # print("Processing for bag")
    bar, bag_dominant_color, bag_ratio, bag_rgb, all_bag_colors = segment_and_cluster(args.bag_segments, cropped_bag,
                                                                                      args.bag_cluster_size)
    sorted_bag_colors = get_sorted_colors(all_bag_colors)
    # print("Sorted bag colors : ", sorted_bag_colors)
    # print("Processing for logo")

    if cropped_logo is not None:
        cropped_logo = np.array(cropped_logo)  # converting pil image to numpy array

        # Get logo size
        height, width, _ = cropped_logo.shape
        logo_size = height * width
        print("Logo size : ", logo_size)

        # Segment and cluster logo colors
        bar, logo_dominant_color, logo_ratio, logo_rgb, all_logo_colors = segment_and_cluster(args.logo_segments,
                                                                                              cropped_logo,
                                                                                              args.logo_cluster_size)

        # Calculate contrast ratio of logo to bag
        contrast_ratio = (calculate_relative_luminance(logo_rgb) + 0.05) / (
                    calculate_relative_luminance(bag_rgb) + 0.05)
        print("Contrast ratio is : ", contrast_ratio)

        # Calculate Logo Conspicuousness
        relative_size = logo_size / bag_size
        logo_conspicuousness = contrast_ratio * relative_size
        print("Logo Conspicuousness : ", logo_conspicuousness)

    else:
        logo_size = 0.0
        contrast_ratio = 0.0
        logo_conspicuousness = 0.0

    # Find total number of colors in the bag
    total_colors = len(sorted_bag_colors)
    print("Total colors are : ", total_colors)

    # Calculate Entropy of the bag
    total_sum = 0

    for color in sorted_bag_colors:
        if color[1] >= 0.05:
            total_sum += color[1]

    entropy = -1 * (total_sum * math.log(total_sum))
    print("Entropy of handbag is : ", entropy)

    return pd.Series(
        (logo_size, round(contrast_ratio, 3), round(logo_conspicuousness, 3), total_colors, round(entropy, 3)))


def main(args):

    item_details = pd.DataFrame(columns=['Seller', 'Item ID', 'category', 'brand', 'Image_path'])
    dir_names = os.listdir(args.dir_path)
    print("Total sellers : ", len(dir_names))
    for seller in dir_names:
        if not seller.startswith('seller_'):
            continue

        product_info = pd.read_csv(args.dir_path + seller + '/ProductInfo.csv', usecols=["Item ID", "category", "brand"])
        product_info['Seller'] = seller

        # Get the image names and add to the dataframe
        product_info['Image_path'] = ''
        img_fnames = os.listdir(args.dir_path + seller + '/Images')
        for idx in range(len(img_fnames)):
            img_path = args.dir_path + seller + '/Images/' + img_fnames[idx]
            img_id = int(img_fnames[idx].split('-')[0])
            img_fnames[idx] = [img_path, img_id]

        img_ids, img_names = map(list, zip(*img_fnames))
        product_info['Image_path'] = product_info['Item ID'].replace(to_replace=img_names, value=img_ids)

        item_details = pd.concat([item_details, product_info], ignore_index=True)

    # Removing the rows that doesn't contain bags
    item_details = item_details[item_details['category'].str.match('Bags')]

    new_cols = ['Logo Size', 'Logo Contrast', 'Logo Conspicuousness', 'Number of colors', 'Color Entropy']
    for col in new_cols:
        item_details[col] = None

    item_details[new_cols] = item_details['Image_path'].apply(process_images)

    # Write to csv
    item_details.to_csv("bag_details.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--dir_path", type=str, default='Group4/Sellers/', help="")
    parser.add_argument("--bag_segments", type=int, default=600, help="")
    parser.add_argument("--bag_cluster_size", type=int, default=10, help="")
    parser.add_argument("--logo_segments", type=int, default=300, help="")
    parser.add_argument("--logo_cluster_size", type=int, default=3, help="")
    args = parser.parse_args()
    main(args)
