import tensorflow as tf
import matplotlib.pyplot as plt
def mp(input,w,thr):
    ws=tf.reduce_sum(tf.multiply(input,w))
    o=tf.cond(ws>=thr,lambda:1.0,lambda:0.0)
    return o
input=tf.constant([0.5,0.3,0.8],dtype=tf.float32)
w=tf.constant([0.2,0.4,0.6],dtype=tf.float32)
thr=tf.constant(0.7,dtype=tf.float32)
original_output=mp(input,w,thr)
print("Original Output:",original_output)

