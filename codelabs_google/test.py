#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session() # It will print some warnings here.
print(sess.run(hello))

