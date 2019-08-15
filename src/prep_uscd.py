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
    try:
        outlier = [[[eval(xx.replace(":", ",")) for xx in re.findall("\d*:\d*", x)]
                    for x in open(pform(path, folder)).read().split("\n")
                    if re.search("\d*:\d*", x)!=None]
                   for folder in os.listdir(path)
                   if ".m" in folder][0]
    except IndexError:
        # no outlier file
        outlier= False
        print("no outliers")

    output = []
    labels = []
    tracker=0
    for idx, folder in enumerate(sorted(os.listdir(path))):
        if searchword in folder and "_gt" not in folder:
            #print("outlier:",outlier[idx-tracker])
            for nr, i in enumerate(sorted(os.listdir(pform(path, folder)))):
                #print(folder,i)
                if i[-4:]==".tif":
                    try:
                        output.append(np.asarray(Image.open(pform(path, folder, "/"+i)).resize(size=size)))

                        if outlier != False and any(nr >= x[0] and nr<=x[1] for x in outlier[idx-tracker]):
                            labels.append(1)
                        else:
                            labels.append(0)
                    except OSError:
                        print("corrupted file: ", path, folder, i)
        else:
            #print("folder:", folder)
            tracker+=1



    return np.expand_dims(np.asarray(output),-1), np.asarray(labels)


path1_train = "./data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
path1_test = "./data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"
path2_train = "./data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
path2_test = "./data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test"



train_x, train_y =  get_data(path1_train, "Train", size=[192,128])
np.savez_compressed("./data/ucsd1_train_x", train_x)
np.savez_compressed("./data/ucsd1_train_y", train_y)

test_x, test_y =  get_data(path1_test, "Test")
np.savez_compressed("./data/ucsd1_test_x", test_x)
np.savez_compressed("./data/ucsd1_test_y", test_y)



train_x, train_y =  get_data(path2_train, "Train")
np.savez_compressed("./data/ucsd2_train_x", train_x)
np.savez_compressed("./data/ucsd2_train_y", train_y)

test_x, test_y =  get_data(path2_test, "Test")
np.savez_compressed("./data/ucsd2_test_x", test_x)
np.savez_compressed("./data/ucsd2_test_y", test_y)
