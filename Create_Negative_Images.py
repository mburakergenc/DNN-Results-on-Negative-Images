"""
The dataset is already in the repo, but here is a small script that will
let you to create negative images from the regular ones.
"""
from PIL import Image
import numpy as np
import PIL.ImageOps
import glob


def create_negative_images(images):
    for img in glob.glob(images):
        im = Image.open(img)
        neg_image = PIL.ImageOps.invert(im)
        neg_image.save(img)

create_negative_images("mnist/test-images/*.jpg")
create_negative_images("mnist/train-images/*.jpg")

