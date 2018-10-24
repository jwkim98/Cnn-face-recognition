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

    image_path = tf.placeholder(tf.string, shape = [])

    image_string = tf.read_file(image_path)
    #puts RGB images (3 channels)
    image_decoded = tf.image.decode_jpeg(image_string, channels = 3)
    #this retuns image of (size.x * size.y * 3)
    image_resized = tf.image.resize_images(image_decoded, size)
    #saves the data as tuple (image, indexNum)

    for directory in directory_list:
        #for each file in subdirectory
        print("index {}: {}".format(index, directory))

        if os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.endswith(".jpg"):
                    #full path of each fil
                    filepath = os.path.join(directory,filename)
                    image = sess.run(image_resized, feed_dict = {image_path : filepath})
                    np.save(destination+ str(filenum) +'.npy', (image, index))
                    filenum += 1
            index += 1


print("saving the data...")

with tf.Session() as sess:
    
    save_images(sess, base_directory = 'face_recognition/', destination = 'formatted_images_face/', size = [256, 256])

