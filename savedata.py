import tensorflow as tf
import numpy as np
import random
import os

# directory :list of file directories
def save_images(sess, base_directory, destination, size = [64, 64]):

    #All sub directories in base directory
    sub_directories = os.listdir(base_directory)

    #full directory path of each subdirectories in base directory
    directory_list = [os.path.join(base_directory , x) for x in sub_directories]

    index = 0
    filenum = 0;
    for directory in directory_list:
        #for each file in subdirectory
        print("index {}: {}".format(index, directory))

        for filename in os.listdir(directory):
            if filename.endswith(".jpg"):
                #full path of each file
                image_string = tf.read_file(os.path.join(directory, filename))
                #puts RGB images (3 channels)
                image_decoded = tf.image.decode_jpeg(image_string, channels = 3)
                #this retuns image of (size.x * size.y * 3)
                image_resized = tf.image.resize_images(image_decoded, size)
                #saves the data as tuple (image, indexNum)

                image_resized = sess.run(image_resized)
                np.save(destination+ str(filenum) +'.npy', (image_resized, index))
                filenum += 1
        index += 1


print("saving the data...")

with tf.Session() as sess:
    
    save_images(sess, base_directory = 'images/', destination = 'formatted_images/', size = [256, 256])

