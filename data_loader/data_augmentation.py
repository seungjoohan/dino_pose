from PIL import Image, ImageOps
import numpy as np
import random
import cv2
from torchvision import transforms
import math
from enum import Enum

class CocoPart(Enum):
    Top = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    Nose = 14
    REye = 15
    REar = 16
    LEye = 17
    LEar = 18
    Spine = 19
    RFinger = 20
    RToe = 21
    LFinger = 22
    LToe = 23
    Background = 24  # Background is not used

"""
TODO: background swap - if not exist, generate on the fly using SAM and save generated seg map
OR use a light model to do generate seg map on the fly?
COMPUTATION TIME/MEMORY VS STORAGE
"""

"""
For all data augmentation functions:
image: PIL Image
keypoints: array of shape (num_keypoints, 3) [x, y, visibility]
z_coords: array of shape (num_keypoints)
"""

def pre_crop_image(image, keypoints):
    """
    Cropping image based on longest side of bounding box
    """
    width, height = image.size
    x_coords = keypoints[:, 0]
    y_coords = keypoints[:, 1]

    # Compute extent of bounding box using keypoints
    x_extent = x_coords[x_coords > 0].max() - x_coords[x_coords > 0].min()
    y_extent = y_coords[y_coords > 0].max() - y_coords[y_coords > 0].min()

    # Compute the new extent using longest side of bounding box
    new_extent = 3 * np.max((x_extent, y_extent))
    x_pad = (new_extent - x_extent) / 2.
    y_pad = (new_extent - y_extent) / 2.

    x_new_min = int(np.max((x_coords[x_coords>0].min() - x_pad, 0)))
    x_new_max = int(np.min((x_coords[x_coords>0].max() + x_pad, width)))
    y_new_min = int(np.max((y_coords[y_coords>0].min() - y_pad, 0)))
    y_new_max = int(np.min((y_coords[y_coords>0].max() + y_pad, height)))

    # Crop image
    aug_img, aug_kps = pose_crop(image, keypoints, x_new_min, y_new_min, x_new_max - x_new_min, y_new_max - y_new_min)
    

    return aug_img, aug_kps
    

def pose_crop(image, keypoints, x_min, y_min, target_width, target_height):
    """
    Crops image and keypoints to target dimensions
    """
    resized = image.crop((x_min, y_min, x_min + target_width, y_min + target_height))

    new_keypoints = np.zeros((keypoints.shape[0], 3))

    # Adjust keypoints
    for i, point in enumerate(keypoints):
        # check if point is included within the cropped image
        if point[0] < x_min or point[0] > x_min + target_width or point[1] < y_min or point[1] > y_min + target_height:
            new_keypoints[i, :] = [0, 0, 0]
        else:
            new_keypoints[i, :] = [point[0] - x_min, point[1] - y_min, point[2]]
        
    return resized, new_keypoints

def pose_random_scale(image, keypoints, z_coords, config_preproc):
    """
    Randomly scales image and keypoints
    """
    scalew = np.random.uniform(config_preproc["random_resize_min"], config_preproc["random_resize_max"])
    scaleh = np.random.uniform(config_preproc["random_resize_min"], config_preproc["random_resize_max"])
    new_img = image.resize((int(image.size[0] * scalew), int(image.size[1] * scaleh)))
    
    new_z_coords = np.zeros((z_coords.shape))
    new_keypoints = np.zeros((keypoints.shape[0], 3))
    for i, point in enumerate(keypoints):
        new_keypoints[i, :] = [point[0] * scalew + 0.5, point[1] * scaleh + 0.5, point[2]]
        new_z = z_coords * np.sqrt(scalew * scaleh)

    return new_img, new_keypoints, new_z

def pose_rotation(image, keypoints, config_preproc):
    """
    Randomly rotates image and keypoints
    """
    deg = random.uniform(config_preproc["rotate_min_degree"], config_preproc["rotate_max_degree"])
    rot_img = image.rotate(deg)

    new_keypoints = np.zeros((keypoints.shape[0], 3))
    for i, point in enumerate(keypoints):
        new_keypoints[i, :] = _rotate_coord(image.size, (0, 0), point, deg)

    return rot_img, new_keypoints

def pose_flip(image, keypoints, z_coords):
    """
    Randomly flips image and keypoints horizontally
    """
    r = np.random.random()
    if r < 0.5:
        return image, keypoints, z_coords
    else:
        flip = image.transpose(Image.FLIP_LEFT_RIGHT)
        new_keypoints, new_z = _flip_coord(image.size, keypoints, z_coords)
        return flip, new_keypoints, new_z

def pose_resize_shortestedge(image, keypoints, z_coords, target_size, processor):
    """
    Resizes image and keypoints to shortest edge of processor.shortest_edge
    """
    # adjust image based on shortest edge
    scale = float(target_size) / float(min(image.size))
    if image.size[1] < image.size[0]:
        newh, neww = target_size, int(scale * image.size[0] + 0.5)
    else:
        newh, neww = int(scale * image.size[1] + 0.5), target_size
    resized_img = image.resize((neww, newh))
    model_input_size = (processor.crop_size['width'], processor.crop_size['height'])

    # add padding if result image is smaller than model input size to make it at least model input size
    pw = ph = 0
    if neww < model_input_size[0] or newh < model_input_size[1]:
        pw = max(0, (model_input_size[0] - neww) // 2)
        ph = max(0, (model_input_size[1] - newh) // 2)
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        new_size = (max(neww, model_input_size[0]), max(newh, model_input_size[1]))
        resized_img = ImageOps.pad(resized_img, new_size, color=color)
    
    # adjust keypoints
    new_keypoints = np.zeros((keypoints.shape[0], 3))
    for i, point in enumerate(keypoints):
        new_keypoints[i, :] = [(point[0] * scale + 0.5) + pw, (point[1] * scale + 0.5) + ph, point[2]]
    new_z = z_coords * scale
    return resized_img, new_keypoints, new_z

def random_occultation(image):
    """
    Randomly occludes image. Doesn't affect keypoints.
    """
    max_occultation_ratio = 0.5
    occultation_prob = 0.3
    if np.random.rand() < occultation_prob:
        return image
    
    # randomly select a patch to occlude
    x_width = max_occultation_ratio * np.random.rand()
    x_start = int((1 - x_width) * np.random.rand() * image.size[0])
    x_end = int(x_start + x_width * image.size[0])
    y_width = max_occultation_ratio * np.random.rand()
    y_start = int((1 - y_width) * np.random.rand() * image.size[1])
    y_end = int(y_start + y_width * image.size[1])

    # occlude patch
    img_arr = np.array(image)
    img_arr[y_start:y_end, x_start:x_end, :] = 0
    occulted_image = Image.fromarray(img_arr)

    return occulted_image

def _flip_coord(shape, keypoints, z_coords):
    """
    Flips the coordinates of the keypoints horizontally
    """
    flipped_list = [CocoPart.Top, CocoPart.Neck, 
                 CocoPart.LShoulder, CocoPart.LElbow, CocoPart.LWrist, 
                 CocoPart.RShoulder, CocoPart.RElbow, CocoPart.RWrist,
                 CocoPart.LHip, CocoPart.LKnee, CocoPart.LAnkle, 
                 CocoPart.RHip, CocoPart.RKnee, CocoPart.RAnkle,
                 CocoPart.Nose, 
                 CocoPart.LEye, CocoPart.LEar, 
                 CocoPart.REye, CocoPart.REar, 
                 CocoPart.Spine,
                 CocoPart.LFinger, CocoPart.LToe, 
                 CocoPart.RFinger, CocoPart.RToe]
    new_keypoints = np.zeros((keypoints.shape[0], 3))
    new_z = np.zeros((z_coords.shape[0]))
    for i, part in enumerate(flipped_list):
        point = keypoints[part.value]
        new_keypoints[i, :] = [shape[0]-point[0], point[1], point[2]]
        new_z[i] = z_coords[part.value]
    return new_keypoints, new_z

def _rotate_coord(shape, newxy, point, angle):
    """
    Rotates the coordinates of the keypoints.
    """
    angle = -1 * angle / 180.0 * math.pi
    ox, oy = shape
    px, py, v = point
    ox /= 2
    oy /= 2
    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    new_x, new_y = newxy
    qx += ox - new_x
    qy += oy - new_y
    return (qx + 0.5), (qy + 0.5), v



if __name__ == "__main__":
    from pycocotools.coco import COCO
    import sys
    import os
    # Add the project root directory to the Python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config.config import get_default_configs
    import matplotlib.pyplot as plt
    from transformers import AutoImageProcessor

    # load annotation
    config_dataset, config_training, config_preproc, config_model = get_default_configs()
    annot_path = config_dataset["train_annotation_json"]
    coco = COCO(annot_path)
    img_ids = coco.getImgIds()

    rand_img_id = np.random.randint(0, len(img_ids)-1)
    img_ann = coco.loadAnns(coco.getAnnIds(imgIds=img_ids[rand_img_id]))[0]
    img_path = coco.loadImgs(img_ids[rand_img_id])[0]["file_name"]
    img = Image.open(os.path.join(config_dataset["train_images_dir"], img_path))
    kps = np.array(img_ann["keypoints"])
    kps = kps.reshape(-1, 3)
    z_coords = np.array(img_ann["keypoints_z"])
    # visualize keypoints
    print("Showing original image")
    plt.imshow(img)
    plt.scatter(kps[:, 0], kps[:, 1], c="red", s=10)
    plt.show()

    # # pre-crop image
    # print("Showing pre-cropped image")
    # aug_img, aug_kps = pre_crop_image(img, kps)
    # plt.imshow(aug_img) 
    # plt.scatter(aug_kps[:, 0], aug_kps[:, 1], c="red", s=10)
    # plt.show()

    # # random scale
    # print("Showing random scaled image")
    # aug_img, aug_kps, aug_z = pose_random_scale(img, kps, z_coords)
    # plt.imshow(aug_img)
    # plt.scatter(aug_kps[:, 0], aug_kps[:, 1], c="red", s=10)
    # plt.show()

    # # random rotation
    # print("Showing random rotated image")
    # aug_img, aug_kps = pose_rotation(img, kps, config_preproc)
    # plt.imshow(aug_img)
    # plt.scatter(aug_kps[:, 0], aug_kps[:, 1], c="red", s=10)
    # plt.show()
    
    # # random flip
    # print("Showing flipped image")
    # aug_img, aug_kps, aug_z = pose_flip(img, kps, z_coords)
    # plt.imshow(aug_img)
    # plt.scatter(aug_kps[:, 0], aug_kps[:, 1], c="red", s=10)
    # plt.show()

    # # resize shortest edge
    # print("Showing resized shortest edge image")
    # processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    # aug_img, aug_kps, aug_z = pose_resize_shortestedge(img, kps, z_coords, 200, processor)
    # print(aug_img.size)
    # plt.imshow(aug_img)
    # plt.scatter(aug_kps[:, 0], aug_kps[:, 1], c="red", s=10)
    # plt.show()

    # random occultation
    print("Showing random occulted image")
    occulted_img = random_occultation(img)
    plt.imshow(occulted_img)
    plt.show()