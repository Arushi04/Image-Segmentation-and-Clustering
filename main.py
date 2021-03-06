import argparse
import pandas as pd
import os
from math import log
import numpy as np
from PIL import Image as Img
import logging

from vision import crop_image, get_logo
from utils import get_sorted_colors, calculate_relative_luminance, remove_glare
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
    logging.info(f"fetching info for image {img_path}")
    if not os.path.isfile(img_path):
        logging.info("\timage not found")
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

    # Removing glares from the image
    cropped_bag = remove_glare(cropped_bag)

    # Segmentation of the image into superpixels, taking average of superpixels and then doing clustering
    color_result = {}
    rgb_colors = []
    for i in range(1, args.bag_cluster_size):
        # print("Processing for bag")
        bar, bag_dominant_color, bag_ratio, bag_rgb, all_bag_colors = segment_and_cluster(args.bag_segments, cropped_bag, i)
        sorted_bag_colors = get_sorted_colors(all_bag_colors)
        color_result[i] = sorted_bag_colors
        rgb_colors.append(bag_rgb)

    #logging.info(f"\tbag rgb colors: {rgb_colors}")
    logging.info(f"\tcolors: {color_result}")

    if cropped_logo is not None:
        cropped_logo = np.array(cropped_logo)  # converting pil image to numpy array

        # Get logo size
        height, width, _ = cropped_logo.shape
        logo_size = height * width

        # Segment and cluster logo colors
        bar, logo_dominant_color, logo_ratio, logo_rgb, all_logo_colors = segment_and_cluster(args.logo_segments,
                                                                                              cropped_logo,
                                                                                              args.logo_cluster_size)

        # Calculate contrast ratio of logo to bag
        contrast_ratio = (calculate_relative_luminance(logo_rgb) + 0.05) / (
                calculate_relative_luminance(bag_rgb) + 0.05)

        # Calculate Logo Conspicuousness
        relative_size = logo_size / bag_size
        logo_conspicuousness = contrast_ratio * relative_size

    else:
        logo_size = 0.0
        contrast_ratio = 0.0
        logo_conspicuousness = 0.0

    # Find total number of colors and entropy of the bag
    k2 = color_result[2]
    k3 = color_result[3]
    k2_dominant_ratio = k2[0][1]
    print("k2_dominant_ratio : ", k2_dominant_ratio)
    if k2_dominant_ratio >= 0.9:
        total_colors = 1
        entropy = 0
    elif k2_dominant_ratio < 0.9 and k2_dominant_ratio >= 0.6:
        total_colors = 2
        entropy = -1 * ((k2_dominant_ratio * log(k2_dominant_ratio, 2)) + (k2[1][1] * log(k2[1][1], 2)))
    else:
        total_colors = 3
        entropy = -1 * ((k3[0][1] * log(k3[0][1], 2)) + (k3[1][1] * log(k3[1][1], 2)) + (k3[2][1] * log(k3[2][1], 2)))

    contrast_ratio = round(contrast_ratio, 3)
    logo_conspicuousness = round(logo_conspicuousness, 3)
    entropy = round(entropy, 3)

    logging.info(f"\tlogo_size: {logo_size}, contrast_ratio: {contrast_ratio}, "
                 f"logo_conspicuousness: {logo_conspicuousness}, entropy:{entropy}")

    return_output = (logo_size,
                     round(contrast_ratio, 3),
                     round(logo_conspicuousness, 3),
                     int(total_colors), round(entropy, 3))

    for i in range(1, args.bag_cluster_size):
        colors, ratios = list(zip(*color_result[i]))
        ratios = [round(val, 2) for val in ratios]
        return_output = return_output + (colors, ratios)

    return pd.Series(return_output)


def main(args):
    item_details = pd.DataFrame(columns=['Seller', 'Item ID', 'category', 'brand', 'color', 'Image_path'])
    dir_names = os.listdir(args.dir_path)
    logging.info(f"Total sellers : {len(dir_names)}")
    for seller in dir_names:
        if not seller.startswith('seller_'):
            continue

        product_info = pd.read_csv(args.dir_path + seller + '/ProductInfo.csv',
                                   usecols=["Item ID", "category", "brand", 'color'])
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
    for i in range(1, args.bag_cluster_size):
        new_cols = new_cols + ['color_k%d' % i, 'color_ratio_k%d' % i]

    for col in new_cols:
        item_details[col] = None

    #item_details = item_details.head(5)
    item_details[new_cols] = item_details['Image_path'].apply(process_images)

    # Write to csv
    item_details.to_csv("bag_details_test.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--dir_path", type=str, default='Group4/Sellers/', help="")
    parser.add_argument("--bag_segments", type=int, default=600, help="")
    parser.add_argument("--bag_cluster_size", type=int, default=4, help="")
    parser.add_argument("--logo_segments", type=int, default=300, help="")
    parser.add_argument("--logo_cluster_size", type=int, default=2, help="")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    main(args)
