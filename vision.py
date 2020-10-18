import io
from google.cloud import vision
from google.oauth2 import service_account

creds = service_account.Credentials.from_service_account_file('My First Project-c1dd0c0c9fd5.json')
client = vision.ImageAnnotatorClient(credentials=creds)


def crop_image(img, img_path):
    '''
    This function takes a PIL image as an input and the image in bytes, detects handbag and returns the vertices of the handbag
    Input : PIL imaage and image in bytes
    Output : cropped handbag and its size
    '''

    with io.open(img_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    height, width = img.size

    objects = client.object_localization(image=image).localized_object_annotations

    cropped_imgs = []
    for object_ in objects:
        if object_.name == 'Bag' or object_.name == 'handbag':
            vects = object_.bounding_poly.normalized_vertices

            x0, x2 = vects[0].x, vects[2].x
            x0, x2 = x0 * width, x2 * width
            y0, y2 = vects[0].y, vects[2].y
            y0, y2 = y0 * height, y2 * height

            vects = [x0, y0, x2 - 1, y2 - 1]

            crop_img = img.crop(vects)
            cropped_imgs.append(crop_img)

    if len(cropped_imgs) > 0:
        cropped_imgs[0].save('cropped.jpg')
        return cropped_imgs[0]
    else:
        img.save('cropped.jpg')
        return img


def get_logo(bag_img, img_path):
    '''
     This functions takes cropped handbag image in PIL format, detects logo and returns the cropped logo
     Input: Cropped handbag image in PIL
     Output: Cropped logo and its size if logo exists else None, 0

     #content = cv2.imencode('.png', np.array(bag_img))[1].tobytes()
     #image = vision.Image(content=content)
     #image = vision.Image(content=cropped_img_bytes)
    '''

    with io.open(img_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Detects logo and returns the size of the logo and vertices
    response = client.logo_detection(image=image)
    annotations = response.logo_annotations
    if len(annotations) != 0:
        for annotation in annotations:
            vects = annotation.bounding_poly.vertices
    else:
        vects = []

    if len(vects) > 0:
        cropped_logo = bag_img.crop([vects[0].x, vects[0].y, vects[2].x - 1, vects[2].y - 1])
        return cropped_logo
    else:
        return None
