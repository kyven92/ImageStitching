from stitching import Stitcher
import cv2 as cv
from pathlib import Path
from stitching.images import Images


from matplotlib import pyplot as plt

import numpy as np

from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher


def get_image_paths(img_set):
    return [str(path.relative_to('.')) for path in Path('imgs').rglob(f'{img_set}*')]


def plot_image(img, figsize_in_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
    
def plot_images(imgs, figsize_in_inches=(5,5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


stitcher = Stitcher(detector="sift", confidence_threshold=0.2)

imgs=["data/b1.jpg", "data/b2.jpg", "data/b3.jpg", "data/b4.jpg"]

images = Images.of(imgs)


medium_imgs = list(images.resize(Images.Resolution.MEDIUM))





finder = FeatureDetector()
features = [finder.detect_features(img) for img in medium_imgs]
keypoints_center_img = finder.draw_keypoints(medium_imgs[1], features[1])


# plot_image(keypoints_center_img, (15,10))


matcher = FeatureMatcher()
matches = matcher.match_features(features)

all_relevant_matches = matcher.draw_matches_matrix(medium_imgs, features, matches, conf_thresh=1, 
                                                   inliers=True, matchColor=(0, 255, 0))

for idx1, idx2, img in all_relevant_matches:
    print(f"Matches Image {idx1+1} to Image {idx2+1}")
    plot_image(img, (20,10))


panorama = stitcher.stitch(imgs)

cv.imwrite('final.png', panorama)