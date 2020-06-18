import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def thresholding(img,mask):
    if(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

data_gen_args = dict(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05,shear_range=0.05, zoom_range=0.05, horizontal_flip=True, fill_mode='nearest')

def trainGenerator(train_path,image_folder,mask_folder,aug_dict = data_gen_args,save_to_dir = None,target_size = (256,256)):    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = 2,
        save_to_dir = save_to_dir,
        save_prefix  = "image",
        seed = 1)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = 2,
        save_to_dir = save_to_dir,
        save_prefix  = "mask",
        seed = 1)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = thresholding(img,mask)
        yield (img,mask)



def testGenerator(test_path,num_image = 30,target_size = (256,256)):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png" % i),as_gray = True)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img


def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)