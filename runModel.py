import tensorflow as tf
import numpy as np
import os
import random

saver = tf.train.import_meta_graph('models/face_model/CnnModel_iter.ckpt-13000.meta')

graph = tf.get_default_graph()
labels = graph.get_tensor_by_name('labels:0')
imageData = graph.get_tensor_by_name('imageData:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')

#result = graph.get_tensor_by_name('result:0')

accuracy = graph.get_tensor_by_name('accuracy:0')

num_categories = 3

def test(sess, test, num_categories):

    image = [x[0] for x in test]
    label = [x[1] for x in test]

    acc = sess.run(accuracy, feed_dict = {imageData: image, labels : label, keep_prob:1.0})

    print("Accuracy: {:.6}%".format(acc*100))



with tf.Session() as sess:

    saver.restore(sess,'models/face_model/CnnModel_iter.ckpt-13000')

    data_list = list()
    acc_list = list()

    dirlist = [name for name in os.listdir("formatted_images_face_test/") if name.endswith('.npy')]

    for filename in dirlist:
        restored = np.load("formatted_images_face_test/" + filename)
        data_list.append(restored)

    label_in = tf.placeholder(tf.int32, shape = [])
    one_hot = tf.one_hot(label_in, num_categories)

    data_list_one_hot = list()
    for element in data_list:
        label_one_hot = sess.run(one_hot, feed_dict = {label_in : element[1]})
        data_list_one_hot.append((element[0], label_one_hot))

    random.shuffle(data_list_one_hot)

    test(sess, data_list_one_hot, num_categories)


