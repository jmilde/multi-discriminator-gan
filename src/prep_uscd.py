import numpy as np
import os
from PIL import Image
from src.util_io import pform

def resize_images(imgs, size=[32, 32]):
    # convert float type to integer
    resized_imgs = np.asarray([np.asarray(Image.fromarray(img).resize(size=size, resample=Image.ANTIALIAS))
                                       for i, img in enumerate(imgs.astype('uint8'))])

    return np.expand_dims(resized_imgs, -1)

def get_data(path, searchword, size=[240,160]):
    output = []
    for folder in os.listdir(path):
        if searchword in folder:
            for i in os.listdir(pform(path, folder)):
                if i[-4:]==".tif":
                    try:
                        output.append(np.asarray(Image.open(pform(path, folder, "/"+i)).resize(size=size)))
                    except OSError:
                        print(path, folder, i)

    return np.expand_dims(np.asarray(output),-1)


path1_train = "./data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
path1_test = "./data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"
path2_train = "./data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
path2_test = "./data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test"

np.savez_compressed("./data/ucsd1_test", get_data(path1_test, "Test"))
np.savez_compressed("./data/ucsd1_train", get_data(path1_train, "Train"))
np.savez_compressed("./data/ucsd2_test", get_data(path2_test, "Test"))
np.savez_compressed("./data/ucsd2_train", get_data(path2_train, "Train"))
