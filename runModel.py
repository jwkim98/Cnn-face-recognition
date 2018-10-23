import tensorflow as tf
import numpy as np


with tf.Session() as sess:
    saver = tf.train.import_meta_graph()
    saver.restore(sess,tf.train.latest_checkpoint('./'))

    graph = tf.get_default_traph()
    labels = graph.get_tensor_by_name('labels:0')
    imageData = graph.get_tensor_by_name('imageData:0')

    feed_dict = {imageData = img}

    result = graph.get_tensor_by_name('result:0')

    prediction = sess.run(result, feed_dict)

    print(prediction)
