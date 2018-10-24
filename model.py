import tensorflow as tf
import numpy as np
import convFunc as my
import matplotlib.pyplot as plt
import random
import gc
import os
import sys

# directory :list of file directories
def importImages(sess, base_directory, size = [64, 64]):
    images = list()

    #All sub directories in base directory
    sub_directories = os.listdir(base_directory)

    #full directory path of each subdirectories in base directory
    directory_list = [os.path.join(base_directory , x) for x in sub_directories]

    index = 0
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
                images.append((image_resized, index))
        index += 1

    return images

def read_formatted_images(sess, dirlist, image_list, img_directory, size):

    start = random.randint(0, len(dirlist) - size)

    parsed_list = dirlist[start: start+size]
    random.shuffle(parsed_list)

    for filename in parsed_list:
        restored = np.load(img_directory + filename)
        #print(restored[1])
        image_list.append(restored)


def createModel(imageData, keep_prob, num_categories):

    conv1 = my.conv_layer(imageData, filter_shape = [11,11,3,96], strides = [1,4,4,1], name = "conv1")
    # conv1 : 64*64*96
    conv1_pool = my.max_pool(conv1, size = [1, 3, 3, 1], stride = [1, 2, 2, 1], name ='conv1_pool')
    # conv1_pool : 32*32*96
    conv1_norm = my.norm(conv1_pool)
    # conv1_norm : 32*32*96

    conv2 = my.conv_layer(conv1_norm, filter_shape = [5, 5, 96, 256], strides = [1,1,1,1], name = "conv2")
    # conv2 : 32*32*256
    conv2_pool = my.max_pool(conv2, size = [1, 3, 3, 1], stride = [1, 2, 2, 1], name = 'conv2_pool')
    # conv2_pool : 16*16*256
    conv2_norm = my.norm(conv2_pool)
    # conv2_norm : 16*16*256

    conv3 = my.conv_layer(conv2_norm, filter_shape = [3, 3, 256, 384], name = "conv3")
    # conv3 : 16*16*384
    conv3_pool = my.max_pool(conv3, size = [1, 3, 3, 1], stride = [1, 2, 2, 1], name = 'conv3_pool')
    # conv3_pool : 8*8*384
    conv3_norm = my.norm(conv3_pool)
    
    #conv4 = my.conv_layer(conv3, filter_shape = [3, 3, 384, 256], name = "conv4")
    # conv4 : 12*12*256

    conv5 = my.conv_layer(conv3_norm, filter_shape = [3, 3, 384, 128], name = "conv5")
    # conv5 : 8*8*128

    #conv5_pool = my.max_pool(conv5, size = [1, 3, 3, 1], stride = [1, 2, 2, 1], name = 'conv5_pool')
    # conv5_pool : 4*4*128

    flat = tf.reshape(conv5, [-1, 8*8*128])

    full0 = my.full_layer(flat, 2048, activation_func = 'relu', keep_prob = keep_prob, name = 'full0')

    full1 = my.full_layer(full0, 1024, activation_func = 'relu', keep_prob = keep_prob, name = 'full1')

    full2 = my.full_layer(full1, 1024, activation_func = 'relu', keep_prob = keep_prob, name = 'full2')

    result = my.full_layer(full2, num_categories, activation_func = 'softmax', name = 'full_softmax')

    return result




num_categories = 3;
labels = tf.placeholder(tf.float32, shape = [None, num_categories], name = 'labels')
imageData = tf.placeholder(tf.float32, shape = [None, 256, 256, 3], name = 'imageData')
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
# in: 227*227*3, out: 64*64*64

model = createModel(imageData, keep_prob, num_categories)
result = tf.identity(model, 'result')

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = result, labels = labels))

#using Adam as optimizer
train_step = tf.train.AdamOptimizer(2e-5).minimize(cross_entropy, name = 'train')

correct_prediction = tf.equal(tf.argmax(result, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

def test(sess, test, num_categories, accuracies):
    #random.shuffle(test)

    image = [x[0] for x in test]
    label = [x[1] for x in test]

    acc = sess.run(accuracy, feed_dict = {imageData: image, labels: label, keep_prob: 1.0})

    accuracies.append(acc*100)
    print("Accuracy: {:.6}%".format(acc*100))

with tf.Session() as sess:
    saver = tf.train.Saver()
    #initialize all variables
    sess.run(tf.global_variables_initializer())

    epoches = 55000
    batch_size = 200
    #put base directory (where the subdirectories are)

    data_list = list()
    acc_list = list()

    dirlist = [name for name in os.listdir("formatted_images_face/") if name.endswith('.npy')]

    for filename in dirlist:
        restored = np.load("formatted_images_face/" + filename)
        data_list.append(restored)

    label_in = tf.placeholder(tf.int32, shape = [])
    one_hot = tf.one_hot(label_in, num_categories)

    data_list_one_hot = list()
    for element in data_list:
        label_one_hot = sess.run(one_hot, feed_dict = {label_in : element[1]})
        data_list_one_hot.append((element[0], label_one_hot))

    random.shuffle(data_list_one_hot)
    print(" training on total data size:{}".format(sys.getsizeof(data_list)))
    test_data = data_list_one_hot[0:200]
    train_data = data_list_one_hot[200: len(data_list_one_hot)]

    for i in range(epoches):

        print("batch: {0}".format(i))

        start = random.randint(0, len(train_data) - batch_size)
        batch = train_data[start: start + batch_size]
        random.shuffle(batch)
        image_batch = [x[0] for x in batch]
        label_batch = [x[1] for x in batch]

        #one-hot encode the label

        sess.run(train_step, feed_dict = {imageData: image_batch, labels: label_batch, keep_prob: 0.5})

        test(sess, test_data, num_categories, acc_list)

        if i%1000 == 0:
            saver.save(sess, 'models/face_model3/iterations/CnnModel_iter.ckpt', global_step = i)
        
        gc.collect()

    save_path = saver.save(sess, "models/face_model3/finalCnnModel.ckpt")
    print("final Model saved in path: %s" %save_path)
plt.plot(acc_list)
plt.xlabel('epoches')
plt.ylabel('accuracy')
plt.show()
plt.savefig('models/face_model3/cnnModel.png')
    

