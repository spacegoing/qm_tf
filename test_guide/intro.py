from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
########## Header ####################

a = tf.constant(3.0)
b = tf.constant(4.0)
total= a+b

print(a)
print(b)
print(total)

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

sess = tf.Session()
d = sess.run({'ac':(a,b), 'total':total})
d = sess.run({'ac':(a,b), 'total':total}, feed_dict={a:4,b:3})
print(d)



