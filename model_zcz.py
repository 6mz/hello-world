# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:29:15 2018

@author: zWX618024
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import ipdb
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
from cjw_op import *




def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels,name):
    pool_size = 2
    pool_size1=2
    #deconv_filter = tf.get_variable(name=name,tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv_filter = tf.get_variable(name=name, shape=[pool_size, pool_size, output_channels, in_channels],initializer=tf.truncated_normal_initializer(stddev=0.02))
    #print(deconv_filter.shape,'deconv_filter.shape')
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size1, pool_size1, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output




def network(input,reuse):  # Unet  use 4 images to test
    conv01 = slim.conv2d(input[0], 32, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv01_1',reuse=reuse)# the first 1 means the 1st layer, the second 1 means the 2nd image
    conv01 = slim.conv2d(conv01, 32, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv01_2',reuse=reuse)
    conv02 = slim.conv2d(input[1], 32, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv02_1',reuse=reuse)# the first 1 means the 1st layer, the second number 2 means the 2nd image
    conv02 = slim.conv2d(conv02, 32, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv02_2',reuse=reuse)
    conv03 = slim.conv2d(input[2], 32, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv03_1',reuse=reuse)
    conv03 = slim.conv2d(conv03, 32, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv03_2',reuse=reuse)
    conv04 = slim.conv2d(input[3], 32, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv04_1',reuse=reuse)
    conv04 = slim.conv2d(conv04, 32, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv04_2',reuse=reuse)
    conv05 = slim.conv2d(input[4], 32, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv05_1',reuse=reuse)
    conv05 = slim.conv2d(conv05, 32, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv05_2',reuse=reuse)
    conv06 = slim.conv2d(input[5], 32, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv06_1',reuse=reuse)
    conv06 = slim.conv2d(conv06, 32, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv06_2',reuse=reuse)
    
    
    print(conv01.shape,'conv01')
    common1=tf.concat([tf.expand_dims(conv01,1),tf.expand_dims(conv02,1),tf.expand_dims(conv03,1),tf.expand_dims(conv04,1),tf.expand_dims(conv05,1),tf.expand_dims(conv06,1)],1)
    #print(common1.shape)
    pool1 = slim.max_pool3d(common1,[6,1,1],[6,1,1],padding='SAME') #pool3d(input,ksize,strides,padding) input.shape()->[batch,in_depth,in_height,in_width,in_channels]
    pool1=tf.squeeze(pool1,[1])
    pool11=conv01
    #print(pool11.shape,'pool11')
    pool12=conv02
    pool13=conv03
    pool14=conv04
    pool15=conv05
    pool16=conv06
    #pool12 = slim.max_pool2d(conv02, [1, 1], padding='SAME')
    #pool13 = slim.max_pool2d(conv03, [1, 1], padding='SAME')
    #pool14 = slim.max_pool2d(conv04, [1, 1], padding='SAME')
    
    concat1_1=tf.concat([pool1,pool11],3)
    #print(concat1_1.shape,'concat1_1')
    concat1_2=tf.concat([pool1,pool12],3)
    concat1_3=tf.concat([pool1,pool13],3)
    concat1_4=tf.concat([pool1,pool14],3) #64
    concat1_5=tf.concat([pool1,pool15],3)
    concat1_6=tf.concat([pool1,pool16],3) #64
    
    conv11 = slim.conv2d(concat1_1, 64, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv11_1',reuse=reuse)# the first 1 means the 1st layer, the second 1 means the 2nd image
    #print(conv11.shape,'conv11_0')
    conv11 = slim.conv2d(conv11, 64, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv11_2',reuse=reuse)
    conv12 = slim.conv2d(concat1_2, 64, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv12_1',reuse=reuse)# the first 1 means the 1st layer, the second number 2 means the 2nd image
    conv12 = slim.conv2d(conv12, 64, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv12_2',reuse=reuse)
    conv13 = slim.conv2d(concat1_3, 64, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv13_1',reuse=reuse)
    conv13 = slim.conv2d(conv13, 64, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv13_2',reuse=reuse)
    conv14 = slim.conv2d(concat1_4, 64, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv14_1',reuse=reuse)
    conv14 = slim.conv2d(conv14, 64, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv14_2',reuse=reuse)
    conv15 = slim.conv2d(concat1_5, 64, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv15_1',reuse=reuse)
    conv15 = slim.conv2d(conv15, 64, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv15_2',reuse=reuse)
    conv16 = slim.conv2d(concat1_6, 64, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv16_1',reuse=reuse)
    conv16 = slim.conv2d(conv16, 64, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv16_2',reuse=reuse)
    #print(conv11.shape,'conv11') #32 32 64
    
    
    
    common2=tf.concat([tf.expand_dims(conv11,1),tf.expand_dims(conv12,1),tf.expand_dims(conv13,1),tf.expand_dims(conv14,1),tf.expand_dims(conv15,1),tf.expand_dims(conv16,1)],1)
    pool2 = slim.max_pool3d(common2,[6,1,1],[6,1,1],padding='SAME') #pool3d(input,ksize,strides,padding) input.shape()->[batch,in_depth,in_height,in_width,in_channels]
    #pool2 = slim.max_pool2d(tf.squeeze(pool2,[1]),[1,1],padding='SAME')
    pool2 = tf.squeeze(pool2,[1])
    #print(pool2.shape,'pool2')
    pool21=conv11
    pool22=conv12
    pool23=conv13
    pool24=conv14
    pool25=conv15
    pool26=conv16

    
    concat2_1=tf.concat([pool2,pool21],3)
    #print(concat2_1.shape,'concat2_1')
    concat2_2=tf.concat([pool2,pool22],3)
    concat2_3=tf.concat([pool2,pool23],3)
    concat2_4=tf.concat([pool2,pool24],3) #128
    concat2_5=tf.concat([pool2,pool25],3)
    concat2_6=tf.concat([pool2,pool26],3) #128
    
    conv21 = slim.conv2d(concat2_1, 128, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv21_1',reuse=reuse)# the first 1 means the 1st layer, the second 1 means the 2nd image
    conv21 = slim.conv2d(conv21, 128, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv21_2',reuse=reuse)
    conv22 = slim.conv2d(concat2_2, 128, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv22_1',reuse=reuse)# the first 1 means the 1st layer, the second number 2 means the 2nd image
    conv22 = slim.conv2d(conv22, 128, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv22_2',reuse=reuse)
    conv23 = slim.conv2d(concat2_3, 128, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv23_1',reuse=reuse)
    conv23 = slim.conv2d(conv23, 128, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv23_2',reuse=reuse)
    conv24 = slim.conv2d(concat2_4, 128, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv24_1',reuse=reuse)
    conv24 = slim.conv2d(conv24, 128, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv24_2',reuse=reuse)
    conv25 = slim.conv2d(concat2_5, 128, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv25_1',reuse=reuse)
    conv25 = slim.conv2d(conv25, 128, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv25_2',reuse=reuse)
    conv26 = slim.conv2d(concat2_6, 128, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv26_1',reuse=reuse)
    conv26 = slim.conv2d(conv26, 128, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv26_2',reuse=reuse)
    #print(conv21.shape,'conv21')



    common3=tf.concat([tf.expand_dims(conv21,1),tf.expand_dims(conv22,1),tf.expand_dims(conv23,1),tf.expand_dims(conv24,1),tf.expand_dims(conv25,1),tf.expand_dims(conv26,1)],1)
    pool3 = slim.max_pool3d(common3,[6,1,1],[6,1,1],padding='SAME') #pool3d(input,ksize,strides,padding) input.shape()->[batch,in_depth,in_height,in_width,in_channels]
    pool3 = tf.squeeze(pool3,[1])
    pool31=conv21
    pool32=conv22
    pool33=conv23
    pool34=conv24
    pool35=conv25
    pool36=conv26
    concat3_1 = tf.concat([pool3,pool31],3)
    concat3_2 = tf.concat([pool3,pool32],3)
    concat3_3 = tf.concat([pool3,pool33],3)
    concat3_4 = tf.concat([pool3,pool34],3)   #256 
    concat3_5 = tf.concat([pool3,pool35],3)
    concat3_6 = tf.concat([pool3,pool36],3)   #256 
    conv31 = slim.conv2d(concat3_1, 256, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv31_1',reuse=reuse)# the first 1 means the 1st layer, the second 1 means the 2nd image
    conv31 = slim.conv2d(conv31, 256, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv31_2',reuse=reuse)
    conv32 = slim.conv2d(concat3_2, 256, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv32_1',reuse=reuse)# the first 1 means the 1st layer, the second number 2 means the 2nd image
    conv32 = slim.conv2d(conv32, 256, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv32_2',reuse=reuse)
    conv33 = slim.conv2d(concat3_3, 256, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv33_1',reuse=reuse)
    conv33 = slim.conv2d(conv33, 256, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv33_2',reuse=reuse)
    conv34 = slim.conv2d(concat3_4, 256, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv34_1',reuse=reuse)
    conv34 = slim.conv2d(conv34, 256, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv34_2',reuse=reuse)
    conv35 = slim.conv2d(concat3_5, 256, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv35_1',reuse=reuse)
    conv35 = slim.conv2d(conv35, 256, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv35_2',reuse=reuse)
    conv36 = slim.conv2d(concat3_6, 256, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv36_1',reuse=reuse)
    conv36 = slim.conv2d(conv36, 256, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv36_2',reuse=reuse)
    
    #print(conv31.shape,'conv31')
    
    
    
    common4=tf.concat([tf.expand_dims(conv31,1),tf.expand_dims(conv32,1),tf.expand_dims(conv33,1),tf.expand_dims(conv34,1),tf.expand_dims(conv35,1),tf.expand_dims(conv36,1)],1)
    pool4 = slim.max_pool3d(common4,[6,1,1],[6,1,1],padding='SAME') #pool3d(input,ksize,strides,padding) input.shape()->[batch,in_depth,in_height,in_width,in_channels]
   # pool4 = slim.max_pool2d(tf.squeeze(pool4,[1]),[1,1],padding='SAME')
    pool4 = tf.squeeze(pool4,[1])
    #pool2=tf.reshape(pool1,tf.shape(input1))# tf.shape(input1)=[batch,height,width,channels]
    '''pool41 = slim.max_pool2d(conv31, [1, 1], padding='SAME')#pool2d(value,ksize,strides,padding) #value.shape()->[batch,height,width,channels]
    pool42 = slim.max_pool2d(conv32, [1, 1], padding='SAME')
    pool43 = slim.max_pool2d(conv33, [1, 1], padding='SAME')
    pool44 = slim.max_pool2d(conv34, [1, 1], padding='SAME')
   '''
    pool41=conv31
    pool42=conv32
    pool43=conv33
    pool44=conv34
    pool45=conv35
    pool46=conv36
    concat4_1=tf.concat([pool4,pool41],3)
    concat4_2=tf.concat([pool4,pool42],3)
    concat4_3=tf.concat([pool4,pool43],3)
    concat4_4=tf.concat([pool4,pool44],3) #512
    concat4_5=tf.concat([pool4,pool45],3)
    concat4_6=tf.concat([pool4,pool46],3) #512
    
    conv41 = slim.conv2d(concat4_1, 512, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv41_1',reuse=reuse)# the first 1 means the 1st layer, the second 1 means the 2nd image
    conv41 = slim.conv2d(conv41, 512, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv41_2',reuse=reuse)
    #print(conv41.shape,'conv41')
    conv42 = slim.conv2d(concat4_2, 512, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv42_1',reuse=reuse)# the first 1 means the 1st layer, the second number 2 means the 2nd image
    conv42 = slim.conv2d(conv42, 512, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv42_2',reuse=reuse)
    conv43 = slim.conv2d(concat4_3, 512, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv43_1',reuse=reuse)
    conv43 = slim.conv2d(conv43, 512, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv43_2',reuse=reuse)
    conv44 = slim.conv2d(concat4_4, 512, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv44_1',reuse=reuse)
    conv44 = slim.conv2d(conv44, 512, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv44_2',reuse=reuse)
    conv45 = slim.conv2d(concat4_5, 512, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv45_1',reuse=reuse)
    conv45 = slim.conv2d(conv45, 512, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv45_2',reuse=reuse)
    conv46 = slim.conv2d(concat4_6, 512, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv46_1',reuse=reuse)
    conv46 = slim.conv2d(conv46, 512, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv46_2',reuse=reuse)
    
    
    
    
    common5=tf.concat([tf.expand_dims(conv41,1),tf.expand_dims(conv42,1),tf.expand_dims(conv43,1),tf.expand_dims(conv44,1),tf.expand_dims(conv45,1),tf.expand_dims(conv46,1)],1)
    pool5 = slim.max_pool3d(common5,[6,1,1],[6,1,1],padding='SAME') #pool3d(input,ksize,strides,padding) input.shape()->[batch,in_depth,in_height,in_width,in_channels]
    pool5 = tf.squeeze(pool5,[1])
    concat5_1=tf.concat([pool5,conv41],3)
    concat5_2=tf.concat([pool5,conv42],3)
    concat5_3=tf.concat([pool5,conv43],3)
    concat5_4=tf.concat([pool5,conv44],3) #1024
    concat5_5=tf.concat([pool5,conv45],3)
    concat5_6=tf.concat([pool5,conv46],3) #1024
    
    conv51 = slim.conv2d(concat5_1, 1024, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv51_1',reuse=reuse)# the first 1 means the 1st layer, the second 1 means the 2nd image
    conv51 = slim.conv2d(conv51, 1024, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv51_2',reuse=reuse)
    conv52 = slim.conv2d(concat5_2, 1024, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv52_1',reuse=reuse)# the first 1 means the 1st layer, the second number 2 means the 2nd image
    conv52 = slim.conv2d(conv52, 1024, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv52_2',reuse=reuse)
    conv53 = slim.conv2d(concat5_3, 1024, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv53_1',reuse=reuse)
    conv53 = slim.conv2d(conv53, 1024, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv53_2',reuse=reuse)
    conv54 = slim.conv2d(concat5_4, 1024, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv54_1',reuse=reuse)
    conv54 = slim.conv2d(conv54, 1024, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv54_2',reuse=reuse)    #1024
    conv55 = slim.conv2d(concat5_5, 1024, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv55_1',reuse=reuse)
    conv55 = slim.conv2d(conv55, 1024, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv55_2',reuse=reuse)
    conv56 = slim.conv2d(concat5_6, 1024, [1, 1], padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv56_1',reuse=reuse)
    conv56 = slim.conv2d(conv56, 1024, [4, 4], stride=2,padding='SAME',rate=1, activation_fn=lrelu, scope='g_conv56_2',reuse=reuse)    #1024
    
    #print(conv51.shape,'conv51')


    
    common6=tf.concat([tf.expand_dims(conv51,1),tf.expand_dims(conv52,1),tf.expand_dims(conv53,1),tf.expand_dims(conv54,1),tf.expand_dims(conv55,1),tf.expand_dims(conv56,1)],1)
    pool6 = slim.max_pool3d(common6,[6,1,1],[6,1,1],padding='SAME') #pool3d(input,ksize,strides,padding) input.shape()->[batch,in_depth,in_height,in_width,in_channels]
    pool6 = tf.squeeze(pool6,[1])
    concat6_1=tf.concat([pool6,conv51],3)
    concat6_2=tf.concat([pool6,conv52],3)
    concat6_3=tf.concat([pool6,conv53],3)
    concat6_4=tf.concat([pool6,conv54],3) #2048
    concat6_5=tf.concat([pool6,conv55],3)
    concat6_6=tf.concat([pool6,conv56],3) 
    #print(concat6_3.shape,'concat6_3')
    up61 = upsample_and_concat(concat6_1, conv41, 512, 2048,'up61')
    #print(up61.shape,'up61')
    conv61 = slim.conv2d(up61, 512, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv61_1',reuse=reuse)
    #print(conv61.shape,'conv61_0')
    conv61 = slim.conv2d(conv61, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv61_2',reuse=reuse)
    #print(conv61.shape,'conv61_1')
    up62 = upsample_and_concat(concat6_2, conv42, 512, 2048,'up62')
    conv62 = slim.conv2d(up62, 512, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv62_1',reuse=reuse)
    conv62 = slim.conv2d(conv62, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv62_2',reuse=reuse)
    up63 = upsample_and_concat(concat6_3, conv43, 512, 2048,'up63')
    conv63 = slim.conv2d(up63, 512, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv63_1',reuse=reuse)
    conv63 = slim.conv2d(conv63, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv63_2',reuse=reuse)
    up64 = upsample_and_concat(concat6_4, conv44, 512, 2048,'up64')
    conv64 = slim.conv2d(up64, 512, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv64_1',reuse=reuse)
    conv64 = slim.conv2d(conv64, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv64_2',reuse=reuse) #512
    up65 = upsample_and_concat(concat6_5, conv45, 512, 2048,'up65')
    conv65 = slim.conv2d(up65, 512, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv65_1',reuse=reuse)
    conv65 = slim.conv2d(conv65, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv65_2',reuse=reuse) #512
    up66 = upsample_and_concat(concat6_6, conv46, 512, 2048,'up66')
    conv66 = slim.conv2d(up66, 512, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv66_1',reuse=reuse)
    conv66 = slim.conv2d(conv66, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv66_2',reuse=reuse) #512
    
    
    
    common7=tf.concat([tf.expand_dims(conv61,1),tf.expand_dims(conv62,1),tf.expand_dims(conv63,1),tf.expand_dims(conv64,1),tf.expand_dims(conv65,1),tf.expand_dims(conv66,1)],1)
    pool7 = slim.max_pool3d(common7,[6,1,1],[6,1,1],padding='SAME') #pool3d(input,ksize,strides,padding) input.shape()->[batch,in_depth,in_height,in_width,in_channels]
    pool7 = tf.squeeze(pool7,[1])
    concat7_1=tf.concat([pool7,conv61],3)
    concat7_2=tf.concat([pool7,conv62],3)
    concat7_3=tf.concat([pool7,conv63],3)
    concat7_4=tf.concat([pool7,conv64],3) #1024
    concat7_5=tf.concat([pool7,conv65],3)
    concat7_6=tf.concat([pool7,conv66],3) #102
    up71 = upsample_and_concat(concat7_1, conv31, 256, 1024,'up71')
    conv71 = slim.conv2d(up71, 256, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv71_1',reuse=reuse)
    conv71 = slim.conv2d(conv71, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv71_2',reuse=reuse)
    up72 = upsample_and_concat(concat7_2, conv32, 256, 1024,'up72')
    conv72 = slim.conv2d(up72, 256, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv72_1',reuse=reuse)
    conv72 = slim.conv2d(conv72, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv72_2',reuse=reuse)
    up73 = upsample_and_concat(concat7_3, conv33, 256, 1024,'up73')
    conv73 = slim.conv2d(up73, 256, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv73_1',reuse=reuse)
    conv73 = slim.conv2d(conv73, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv73_2',reuse=reuse)
    up74 = upsample_and_concat(concat7_4, conv34, 256, 1024,'up74')
    conv74 = slim.conv2d(up74, 256, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv74_1',reuse=reuse)
    conv74 = slim.conv2d(conv74, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv74_2',reuse=reuse)
    up75 = upsample_and_concat(concat7_5, conv35, 256, 1024,'up75')
    conv75 = slim.conv2d(up75, 256, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv75_1',reuse=reuse)
    conv75 = slim.conv2d(conv75, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv75_2',reuse=reuse)
    up76 = upsample_and_concat(concat7_6, conv36, 256, 1024,'up76')
    conv76 = slim.conv2d(up76, 256, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv76_1',reuse=reuse)
    conv76 = slim.conv2d(conv76, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv76_2',reuse=reuse)
    
    
    
    common8=tf.concat([tf.expand_dims(conv71,1),tf.expand_dims(conv72,1),tf.expand_dims(conv73,1),tf.expand_dims(conv74,1),tf.expand_dims(conv75,1),tf.expand_dims(conv76,1)],1)
    pool8 = slim.max_pool3d(common8,[6,1,1],[6,1,1],padding='SAME') #pool3d(input,ksize,strides,padding) input.shape()->[batch,in_depth,in_height,in_width,in_channels]
    pool8 = tf.squeeze(pool8,[1])
    concat8_1=tf.concat([pool8,conv71],3)
    concat8_2=tf.concat([pool8,conv72],3)
    concat8_3=tf.concat([pool8,conv73],3)
    concat8_4=tf.concat([pool8,conv74],3) #512
    concat8_5=tf.concat([pool8,conv75],3)
    concat8_6=tf.concat([pool8,conv76],3) #512
    up81 = upsample_and_concat(concat8_1, conv21, 128, 512,'up81')
    conv81 = slim.conv2d(up81, 128, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv81_1',reuse=reuse)
    conv81 = slim.conv2d(conv81, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv81_2',reuse=reuse)
    up82 = upsample_and_concat(concat8_2, conv22, 128, 512,'up82')
    conv82 = slim.conv2d(up82, 128, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv82_1',reuse=reuse)
    conv82 = slim.conv2d(conv82, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv82_2',reuse=reuse)
    up83 = upsample_and_concat(concat8_3, conv23, 128, 512,'up83')
    conv83 = slim.conv2d(up83, 128, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv83_1',reuse=reuse)
    conv83 = slim.conv2d(conv83, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv83_2',reuse=reuse)
    up84 = upsample_and_concat(concat8_4, conv24, 128, 512,'up84')
    conv84 = slim.conv2d(up84, 128, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv84_1',reuse=reuse)
    conv84 = slim.conv2d(conv84, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv84_2',reuse=reuse)  #128
    up85 = upsample_and_concat(concat8_5, conv25, 128, 512,'up85')
    conv85 = slim.conv2d(up85, 128, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv85_1',reuse=reuse)
    conv85 = slim.conv2d(conv85, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv85_2',reuse=reuse)  #128
    up86 = upsample_and_concat(concat8_6, conv26, 128, 512,'up86')
    conv86 = slim.conv2d(up86, 128, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv86_1',reuse=reuse)
    conv86 = slim.conv2d(conv86, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv86_2',reuse=reuse)  #128
    
    
    
    common9=tf.concat([tf.expand_dims(conv81,1),tf.expand_dims(conv82,1),tf.expand_dims(conv83,1),tf.expand_dims(conv84,1),tf.expand_dims(conv85,1),tf.expand_dims(conv86,1)],1)
    pool9 = slim.max_pool3d(common9,[6,1,1],[6,1,1],padding='SAME') #pool3d(input,ksize,strides,padding) input.shape()->[batch,in_depth,in_height,in_width,in_channels]
    pool9 = tf.squeeze(pool9,[1])
    concat9_1=tf.concat([pool9,conv81],3)
    concat9_2=tf.concat([pool9,conv82],3)
    concat9_3=tf.concat([pool9,conv83],3)
    concat9_4=tf.concat([pool9,conv84],3) #256
    concat9_5=tf.concat([pool9,conv85],3)
    concat9_6=tf.concat([pool9,conv86],3) #256
    up91 = upsample_and_concat(concat9_1, conv11, 64, 256,'up91')
    conv91 = slim.conv2d(up91, 64, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv91_1',reuse=reuse)
    conv91 = slim.conv2d(conv91, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv91_2',reuse=reuse)
    up92 = upsample_and_concat(concat9_2, conv12, 64, 256,'up92')
    conv92 = slim.conv2d(up92, 64, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv92_1',reuse=reuse)
    conv92 = slim.conv2d(conv92, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv92_2',reuse=reuse)
    up93 = upsample_and_concat(concat9_3, conv13, 64, 256,'up93')
    conv93 = slim.conv2d(up93, 64, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv93_1',reuse=reuse)
    conv93 = slim.conv2d(conv93, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv93_2',reuse=reuse)
    up94 = upsample_and_concat(concat9_4, conv14, 64, 256,'up94')
    conv94 = slim.conv2d(up94, 64, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv94_1',reuse=reuse)
    conv94 = slim.conv2d(conv94, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv94_2',reuse=reuse)#64
    up95 = upsample_and_concat(concat9_5, conv15, 64, 256,'up95')
    conv95 = slim.conv2d(up95, 64, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv95_1',reuse=reuse)
    conv95 = slim.conv2d(conv95, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv95_2',reuse=reuse)
    up96 = upsample_and_concat(concat9_6, conv16, 64, 256,'up96')
    conv96 = slim.conv2d(up96, 64, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv96_1',reuse=reuse)
    conv96 = slim.conv2d(conv96, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv96_2',reuse=reuse)#64
    
    
    
    
    commona=tf.concat([tf.expand_dims(conv91,1),tf.expand_dims(conv92,1),tf.expand_dims(conv93,1),tf.expand_dims(conv94,1),tf.expand_dims(conv95,1),tf.expand_dims(conv96,1)],1)
    poola = slim.max_pool3d(commona,[6,1,1],[6,1,1],padding='SAME') #pool3d(input,ksize,strides,padding) input.shape()->[batch,in_depth,in_height,in_width,in_channels]
    poola = tf.squeeze(poola,[1])
    concata_1=tf.concat([poola,conv91],3)
    concata_2=tf.concat([poola,conv92],3)
    concata_3=tf.concat([poola,conv93],3)
    concata_4=tf.concat([poola,conv94],3) #128
    concata_5=tf.concat([poola,conv95],3)
    concata_6=tf.concat([poola,conv96],3) #128
    upa1 = upsample_and_concat(concata_1, conv01, 32, 128,'upa1')
    conva1 = slim.conv2d(upa1, 32, [1, 1], rate=1, activation_fn=lrelu, scope='g_conva1_1',reuse=reuse)
    conva1 = slim.conv2d(conva1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conva1_2',reuse=reuse)
    upa2 = upsample_and_concat(concata_2, conv02, 32, 128,'upa2')
    conva2 = slim.conv2d(upa2, 32, [1, 1], rate=1, activation_fn=lrelu, scope='g_conva2_1',reuse=reuse)
    conva2 = slim.conv2d(conva2, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conva2_2',reuse=reuse)
    upa3 = upsample_and_concat(concata_3, conv03, 32, 128,'upa3')
    conva3 = slim.conv2d(upa3, 32, [1, 1], rate=1, activation_fn=lrelu, scope='g_conva3_1',reuse=reuse)
    conva3 = slim.conv2d(conva3, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conva3_2',reuse=reuse)
    upa4 = upsample_and_concat(concata_4, conv04, 32, 128,'upa4')
    conva4 = slim.conv2d(upa4, 32, [1, 1], rate=1, activation_fn=lrelu, scope='g_conva4_1',reuse=reuse)
    conva4 = slim.conv2d(conva4, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conva4_2',reuse=reuse)
    upa5 = upsample_and_concat(concata_5, conv05, 32, 128,'upa5')
    conva5 = slim.conv2d(upa5, 32, [1, 1], rate=1, activation_fn=lrelu, scope='g_conva5_1',reuse=reuse)
    conva5 = slim.conv2d(conva5, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conva5_2',reuse=reuse)
    upa6 = upsample_and_concat(concata_6, conv06, 32, 128,'upa6')
    conva6 = slim.conv2d(upa6, 32, [1, 1], rate=1, activation_fn=lrelu, scope='g_conva6_1',reuse=reuse)
    conva6 = slim.conv2d(conva6, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conva6_2',reuse=reuse)
    
    
    commonb=tf.concat([tf.expand_dims(conva1,1),tf.expand_dims(conva2,1),tf.expand_dims(conva3,1),tf.expand_dims(conva4,1),tf.expand_dims(conva5,1),tf.expand_dims(conva6,1)],1)
    poolb = slim.max_pool3d(commonb,[6,1,1],[6,1,1],padding='SAME') #pool3d(input,ksize,strides,padding) input.shape()->[batch,in_depth,in_height,in_width,in_channels]
    poolb = tf.squeeze(poolb,[1])
    #print(poolb.shape,'poolb')
    convb=slim.conv2d(poolb,16,[1,1],rate=1,activation_fn=None,scope='g_convb',reuse=reuse)#
    out=tf.depth_to_space(convb,2)#
    print(out.shape,'out.shape')
    return out



def my_deconv(x1,x2,out_channels,in_channels,name):
    pool_size = 2#?
    stride=2
    x2_shape = tf.shape(x2)
    deconv_filter = tf.get_variable(name=name, shape=[pool_size, pool_size, out_channels, in_channels],initializer=tf.truncated_normal_initializer(stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, [x2_shape[0], x2_shape[1],x2_shape[2], out_channels], strides=[1, stride, stride, 1])
    return deconv



    '''
def my_deconv(x1, x2, output_channels, in_channels,name,reuse):

    #pool_size = 2
    #x2_shape = tf.shape(x2)
    #deconv_filter = tf.get_variable(name=name,shape=[pool_size, pool_size, output_channels, in_channels],initializer=tf.truncated_normal_initializer(stddev=0.02))
    deconv = slim.conv2d_transpose(x1, output_channels, [4, 4], stride=2,padding='SAME', activation_fn=tf.nn.elu, scope=name, reuse=reuse)    #deconv_output.set_shape([None, None, None,output_channels])

    return deconv
'''

def PICNN_V1_1(input,frm_num,reuse=False):  # Unet  use 4 images to test
    with tf.variable_scope("PICNN_V1_1", reuse=reuse) as v1_1:
        n=input
        #print(n[1].shape,'input') 

        #common1=tf.cast([],tf.float32)
        nn1=[None]*frm_num
        common10=[None]*frm_num
        concat1=[None]*frm_num
        for i in range(frm_num):
            nn1[i]=slim.conv2d(n[i],32,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer1_%s'%i,reuse=reuse)
            common10[i]=tf.expand_dims(nn1[i],1,name='common1_%s'%i)  
        common1=tf.concat(common10,axis=1,name='common1')
        #print(common1.shape,'common1')
        pool1=slim.max_pool3d(common1,[6,1,1],[6,1,1],padding='SAME',scope='pool1')
        pool1=tf.squeeze(pool1,[1],name='layer2')
        #print(pool1.shape,'pool1')
        for i in range(frm_num):
            concat1[i]=tf.concat([pool1,nn1[i]],axis=3,name='layer3_%s'%i)
        
        nn2=[None]*frm_num
        nn2_1=[None]*frm_num
        common20=[None]*frm_num    
        concat2=[None]*frm_num
        for i in range(frm_num):
            nn2[i]=slim.conv2d(concat1[i],64,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer4_%s'%i,reuse=reuse)
            nn2_1[i]=slim.conv2d(nn2[i],64,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer5_%s'%i,reuse=reuse)
            common20[i]=tf.expand_dims(nn2_1[i],1)
        common2=tf.concat(common20,axis=1,name='common2')
        #print(common2.shape,'common2')
        pool2=slim.max_pool3d(common2,[6,1,1],[6,1,1],padding='SAME',scope='pool2')
        pool2=tf.squeeze(pool2,[1],name='layer6')
        #print(pool2.shape,'pool2')
        for i in range(frm_num):
            concat2[i]=tf.concat([pool2,nn2_1[i]],axis=3,name='layer7_%s'%i)


        nn3=[None]*frm_num
        nn3_1=[None]*frm_num
        common30=[None]*frm_num    
        concat3=[None]*frm_num
        for i in range(frm_num):
            nn3[i]=slim.conv2d(concat2[i],128,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer8_%s'%i,reuse=reuse) 
            nn3_1[i]=slim.conv2d(nn3[i],128,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer9_%s'%i,reuse=reuse)
            common30=tf.expand_dims(nn3_1[i],1)
        common3=tf.concat(common30,axis=1,name='common3')
        #print(common3.shape,'common3')
        pool3=slim.max_pool3d(common3,[6,1,1],[6,1,1],padding='SAME',scope='pool3')
        pool3=tf.squeeze(pool3,[1],name='layer10')
        #print(pool3.shape,'pool3')
        for i in range(frm_num):
            concat3[i]=tf.concat([pool3,nn3_1[i]],axis=3,name='layer11_%s'%i)
        
        
        
        nn4=[None]*frm_num
        nn4_1=[None]*frm_num
        common40=[None]*frm_num    
        concat4=[None]*frm_num
        for i in range(frm_num):     #conv4
            nn4[i]=slim.conv2d(concat3[i],256,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer12_%s'%i,reuse=reuse)  
            nn4_1[i]=slim.conv2d(nn4[i],256,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer13_%s'%i,reuse=reuse)
            common40[i]=tf.expand_dims(nn4_1[i],1)
        common4=tf.concat(common40,axis=1,name='common4')   
        #print(common4.shape,'common4')
        pool4=slim.max_pool3d(common4,[6,1,1],[6,1,1],padding='SAME',scope='pool4')
        pool4=tf.squeeze(pool4,[1],name='layer14')
        #print(pool4.shape,'pool4')
        for i in range(frm_num):
            concat4[i]=tf.concat([pool4,nn4_1[i]],axis=3,name='layer15_%s'%i) 



        nn5=[None]*frm_num
        nn5_1=[None]*frm_num
        common50=[None]*frm_num
        denn5=[None]*frm_num
        denn5_1=[None]*frm_num
        concat5=[None]*frm_num
        for i in range(frm_num):       #conv5
            nn5[i]=slim.conv2d(concat4[i],384,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer16_%s'%i,reuse=reuse)
            nn5_1[i]=slim.conv2d(nn5[i], 384,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer17_%s'%i,reuse=reuse)
            #print(nn5_1[i].shape,'nn5[i]')
            denn5[i]=slim.conv2d(nn5_1[i], 384,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer18_%s'%i,reuse=reuse)            
            denn5_1[i]=my_deconv(denn5[i],nn5[i],384,384,name='layer19%s'%i) 
            #print(denn5[i].shape,'denn5[i]')
            common50[i]=tf.expand_dims(denn5_1[i],1)
        common5=tf.concat(common50,axis=1,name='common5')
        #print(common5.shape,'common5')
        pool5=slim.max_pool3d(common5,[6,1,1],[6,1,1],padding='SAME',scope='pool5')
        pool5=tf.squeeze(pool5,[1],name='layer20')#256
        for i in range(frm_num):
            concat5[i]=tf.concat([pool5,denn5_1[i],nn5[i]],axis=3,name='layer21_%s'%i) #384*3
            


        back_nn4=[None]*frm_num
        denn4=[None]*frm_num
        back_nn4_1=[None]*frm_num
        decommon40=[None]*frm_num
        concat6=[None]*frm_num
        for i in range(frm_num):     #deconv4
            back_nn4[i]=slim.conv2d(concat5[i],384,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer22_%s'%i,reuse=reuse) 
            back_nn4_1[i]=slim.conv2d(back_nn4[i],384,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer23_%s'%i,reuse=reuse) 
            #print(back_nn4_1[i].shape,'back_nn4[i]')
            denn4[i]=my_deconv(back_nn4_1[i],nn4[i],256,384,name='layer24_%s'%i) 
            #print(denn4[i].shape,'denn4[i]')
            #decommon4=tf.concat([tf.expand_dims(denn4,1),decommon4],axis=1,name='decommon4%s'%i)
            decommon40[i]=tf.expand_dims(denn4[i],1)
        decommon4=tf.concat(decommon40,axis=1,name='decommon4')     
        #print(decommon4.shape,'decommon4')
        depool4=slim.max_pool3d(decommon4,[6,1,1],[6,1,1],padding='SAME',scope='depool4')
        depool4=tf.squeeze(depool4,[1],name='layer25')
        for i in range(frm_num):
            concat6[i]=tf.concat([depool4,denn4[i],nn4[i]],axis=3,name='layer26_%s'%i) #128*3

        back_nn3=[None]*frm_num
        back_nn3_1=[None]*frm_num
        denn3=[None]*frm_num
        decommon30=[None]*frm_num
        concat7=[None]*frm_num
        for i in range(frm_num):     #deconv3
            back_nn3[i]=slim.conv2d(concat6[i],256,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer27_%s'%i,reuse=reuse) 
            back_nn3_1[i]=slim.conv2d(back_nn3[i],256,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.relu,scope='layer28_%s'%i,reuse=reuse)
            denn3[i]=my_deconv(back_nn3_1[i],nn3[i],192,256,name='layer29_%s'%i) 
            #print(back_nn3_1[i].shape,'back_nn3[i]')
            #print(denn3[i].shape,'denn3[i]')
            decommon30[i]=tf.expand_dims(denn3[i],1)
        decommon3=tf.concat(decommon30,axis=1,name='decommon3')
        #print(decommon3.shape,'decommon3')
        depool3=slim.max_pool3d(decommon3,[6,1,1],[6,1,1],padding='SAME',scope='depool3')
        depool3=tf.squeeze(depool3,[1],name='layer30')
        for i in range(frm_num):
            concat7[i]=tf.concat([depool3,denn3[i],nn3[i]],axis=3,name='layer31_%s'%i) #192*3
      
            

        back_nn2=[None]*frm_num
        denn2=[None]*frm_num
        decommon20=[None]*frm_num
        concat8=[None]*frm_num
        for i in range(frm_num):     #deconv2
            back_nn2[i]=slim.conv2d(concat7[i],192,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer32_%s'%i,reuse=reuse) #64
            back_nn2[i]=slim.conv2d(back_nn2[i],192,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer33_%s'%i,reuse=reuse) #64->64
            denn2[i]=my_deconv(back_nn2[i],nn2[i],64,192,name='layer34_%s'%i) 
            #print(back_nn2[i].shape,'back_nn2[i]')
            #print(denn2[i].shape,'denn2[i]')
            decommon20[i]=tf.expand_dims(denn2[i],1)
        decommon2=tf.concat(decommon20,axis=1,name='decommon2')
        #print(decommon2.shape,'decommon2')
        depool2=slim.max_pool3d(decommon2,[6,1,1],[6,1,1],padding='SAME',scope='depool2')
        depool2=tf.squeeze(depool2,[1],name='layer35')
        for i in range(frm_num):
            concat8[i]=tf.concat([depool2,denn2[i],nn2[i]],axis=3,name='layer36_%s'%i) 
               
        back_nn1=[None]*frm_num
        #denn1=[None]*frm_num
        decommon10=[None]*frm_num
        for i in range(frm_num):     #deconv2
            back_nn1[i]=slim.conv2d(concat8[i],64,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer37_%s'%i,reuse=reuse) #64
            back_nn1[i]=slim.conv2d(back_nn1[i],32,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer38_%s'%i,reuse=reuse) #64->64
            decommon10[i]=tf.expand_dims(back_nn1[i],1)
        decommon1=tf.concat(decommon10,axis=1,name='decommon1')
        #print(decommon1.shape,'decommon1')
        depool1=slim.max_pool3d(decommon1,[6,1,1],[6,1,1],padding='SAME',scope='depool1')
        depool1=tf.squeeze(depool1,[1],name='layer39')
        
        
               
        
        
        
        n_final=slim.conv2d(depool1,32,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer40_%s'%i,reuse=reuse) 
        #print(n_final.shape,'n_final')
        out=slim.conv2d(n_final,4,[3,3],stride=1,padding='SAME',activation_fn=None,scope='layer41',reuse=reuse)
        #print(out.shape,'out')
        return out





def PICNN_V1_2(input,frm_num,avr_raw,reuse=False):  # Unet  use 4 images to test
    with tf.variable_scope("PICNN_V1_2", reuse=reuse) as v1_2:
        n=input
        #print(n[1].shape,'input') 

        #common1=tf.cast([],tf.float32)
        nn1=[None]*frm_num
        common10=[None]*frm_num
        concat1=[None]*frm_num
        for i in range(frm_num):
            nn1[i]=slim.conv2d(n[i],32,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer1_%s'%i,reuse=reuse)
            common10[i]=tf.expand_dims(nn1[i],1,name='common1_%s'%i)  
        common1=tf.concat(common10,axis=1,name='common1')
        #print(common1.shape,'common1')
        pool1=slim.max_pool3d(common1,[6,1,1],[6,1,1],padding='SAME',scope='pool1')
        pool1=tf.squeeze(pool1,[1],name='layer2')
        #print(pool1.shape,'pool1')
        for i in range(frm_num):
            concat1[i]=tf.concat([pool1,nn1[i]],axis=3,name='layer3_%s'%i)
        
        nn2=[None]*frm_num
        nn2_1=[None]*frm_num
        common20=[None]*frm_num    
        concat2=[None]*frm_num
        for i in range(frm_num):
            nn2[i]=slim.conv2d(concat1[i],64,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer4_%s'%i,reuse=reuse)
            nn2_1[i]=slim.conv2d(nn2[i],64,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer5_%s'%i,reuse=reuse)
            common20[i]=tf.expand_dims(nn2_1[i],1)
        common2=tf.concat(common20,axis=1,name='common2')
        #print(common2.shape,'common2')
        pool2=slim.max_pool3d(common2,[6,1,1],[6,1,1],padding='SAME',scope='pool2')
        pool2=tf.squeeze(pool2,[1],name='layer6')
        #print(pool2.shape,'pool2')
        for i in range(frm_num):
            concat2[i]=tf.concat([pool2,nn2_1[i]],axis=3,name='layer7_%s'%i)


        nn3=[None]*frm_num
        nn3_1=[None]*frm_num
        common30=[None]*frm_num    
        concat3=[None]*frm_num
        for i in range(frm_num):
            nn3[i]=slim.conv2d(concat2[i],128,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer8_%s'%i,reuse=reuse) 
            nn3_1[i]=slim.conv2d(nn3[i],128,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer9_%s'%i,reuse=reuse)
            common30=tf.expand_dims(nn3_1[i],1)
        common3=tf.concat(common30,axis=1,name='common3')
        #print(common3.shape,'common3')
        pool3=slim.max_pool3d(common3,[6,1,1],[6,1,1],padding='SAME',scope='pool3')
        pool3=tf.squeeze(pool3,[1],name='layer10')
        #print(pool3.shape,'pool3')
        for i in range(frm_num):
            concat3[i]=tf.concat([pool3,nn3_1[i]],axis=3,name='layer11_%s'%i)
        
        
        
        nn4=[None]*frm_num
        nn4_1=[None]*frm_num
        common40=[None]*frm_num    
        concat4=[None]*frm_num
        for i in range(frm_num):     #conv4
            nn4[i]=slim.conv2d(concat3[i],256,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer12_%s'%i,reuse=reuse)  
            nn4_1[i]=slim.conv2d(nn4[i],256,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer13_%s'%i,reuse=reuse)
            common40[i]=tf.expand_dims(nn4_1[i],1)
        common4=tf.concat(common40,axis=1,name='common4')   
        #print(common4.shape,'common4')
        pool4=slim.max_pool3d(common4,[6,1,1],[6,1,1],padding='SAME',scope='pool4')
        pool4=tf.squeeze(pool4,[1],name='layer14')
        #print(pool4.shape,'pool4')
        for i in range(frm_num):
            concat4[i]=tf.concat([pool4,nn4_1[i]],axis=3,name='layer15_%s'%i) 



        nn5=[None]*frm_num
        nn5_1=[None]*frm_num
        common50=[None]*frm_num
        denn5=[None]*frm_num
        denn5_1=[None]*frm_num
        concat5=[None]*frm_num
        for i in range(frm_num):       #conv5
            nn5[i]=slim.conv2d(concat4[i],384,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer16_%s'%i,reuse=reuse)
            nn5_1[i]=slim.conv2d(nn5[i], 384,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer17_%s'%i,reuse=reuse)
            #print(nn5_1[i].shape,'nn5[i]')
            denn5[i]=slim.conv2d(nn5_1[i], 384,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer18_%s'%i,reuse=reuse)            
            denn5_1[i]=my_deconv(denn5[i],nn5[i],384,384,name='layer19%s'%i) 
            #print(denn5[i].shape,'denn5[i]')
            common50[i]=tf.expand_dims(denn5_1[i],1)
        common5=tf.concat(common50,axis=1,name='common5')
        #print(common5.shape,'common5')
        pool5=slim.max_pool3d(common5,[6,1,1],[6,1,1],padding='SAME',scope='pool5')
        pool5=tf.squeeze(pool5,[1],name='layer20')#256
        for i in range(frm_num):
            concat5[i]=tf.concat([pool5,denn5_1[i],nn5[i]],axis=3,name='layer21_%s'%i) #384*3
            


        back_nn4=[None]*frm_num
        denn4=[None]*frm_num
        back_nn4_1=[None]*frm_num
        decommon40=[None]*frm_num
        concat6=[None]*frm_num
        for i in range(frm_num):     #deconv4
            back_nn4[i]=slim.conv2d(concat5[i],384,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer22_%s'%i,reuse=reuse) 
            back_nn4_1[i]=slim.conv2d(back_nn4[i],384,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer23_%s'%i,reuse=reuse) 
            #print(back_nn4_1[i].shape,'back_nn4[i]')
            denn4[i]=my_deconv(back_nn4_1[i],nn4[i],256,384,name='layer24_%s'%i) 
            #print(denn4[i].shape,'denn4[i]')
            #decommon4=tf.concat([tf.expand_dims(denn4,1),decommon4],axis=1,name='decommon4%s'%i)
            decommon40[i]=tf.expand_dims(denn4[i],1)
        decommon4=tf.concat(decommon40,axis=1,name='decommon4')     
        #print(decommon4.shape,'decommon4')
        depool4=slim.max_pool3d(decommon4,[6,1,1],[6,1,1],padding='SAME',scope='depool4')
        depool4=tf.squeeze(depool4,[1],name='layer25')
        for i in range(frm_num):
            concat6[i]=tf.concat([depool4,denn4[i],nn4[i]],axis=3,name='layer26_%s'%i) #128*3

        back_nn3=[None]*frm_num
        back_nn3_1=[None]*frm_num
        denn3=[None]*frm_num
        decommon30=[None]*frm_num
        concat7=[None]*frm_num
        for i in range(frm_num):     #deconv3
            back_nn3[i]=slim.conv2d(concat6[i],256,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer27_%s'%i,reuse=reuse) 
            back_nn3_1[i]=slim.conv2d(back_nn3[i],256,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.relu,scope='layer28_%s'%i,reuse=reuse)
            denn3[i]=my_deconv(back_nn3_1[i],nn3[i],192,256,name='layer29_%s'%i) 
            #print(back_nn3_1[i].shape,'back_nn3[i]')
            #print(denn3[i].shape,'denn3[i]')
            decommon30[i]=tf.expand_dims(denn3[i],1)
        decommon3=tf.concat(decommon30,axis=1,name='decommon3')
        #print(decommon3.shape,'decommon3')
        depool3=slim.max_pool3d(decommon3,[6,1,1],[6,1,1],padding='SAME',scope='depool3')
        depool3=tf.squeeze(depool3,[1],name='layer30')
        for i in range(frm_num):
            concat7[i]=tf.concat([depool3,denn3[i],nn3[i]],axis=3,name='layer31_%s'%i) #192*3
      
            

        back_nn2=[None]*frm_num
        denn2=[None]*frm_num
        decommon20=[None]*frm_num
        concat8=[None]*frm_num
        for i in range(frm_num):     #deconv2
            back_nn2[i]=slim.conv2d(concat7[i],192,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer32_%s'%i,reuse=reuse) #64
            back_nn2[i]=slim.conv2d(back_nn2[i],192,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer33_%s'%i,reuse=reuse) #64->64
            denn2[i]=my_deconv(back_nn2[i],nn2[i],64,192,name='layer34_%s'%i) 
            #print(back_nn2[i].shape,'back_nn2[i]')
            #print(denn2[i].shape,'denn2[i]')
            decommon20[i]=tf.expand_dims(denn2[i],1)
        decommon2=tf.concat(decommon20,axis=1,name='decommon2')
        #print(decommon2.shape,'decommon2')
        depool2=slim.max_pool3d(decommon2,[6,1,1],[6,1,1],padding='SAME',scope='depool2')
        depool2=tf.squeeze(depool2,[1],name='layer35')
        for i in range(frm_num):
            concat8[i]=tf.concat([depool2,denn2[i],nn2[i]],axis=3,name='layer36_%s'%i) 
               
        back_nn1=[None]*frm_num
        #denn1=[None]*frm_num
        decommon10=[None]*frm_num
        for i in range(frm_num):     #deconv2
            back_nn1[i]=slim.conv2d(concat8[i],64,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer37_%s'%i,reuse=reuse) #64
            back_nn1[i]=slim.conv2d(back_nn1[i],32,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer38_%s'%i,reuse=reuse) #64->64
            decommon10[i]=tf.expand_dims(back_nn1[i],1)
        decommon1=tf.concat(decommon10,axis=1,name='decommon1')
        #print(decommon1.shape,'decommon1')
        depool1=slim.max_pool3d(decommon1,[6,1,1],[6,1,1],padding='SAME',scope='depool1')
        depool1=tf.squeeze(depool1,[1],name='layer39')
        
        
               
        
        
        
        n_final=slim.conv2d(depool1,32,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer40_%s'%i,reuse=reuse) 
        #print(n_final.shape,'n_final')
        out=slim.conv2d(n_final,4,[3,3],stride=1,padding='SAME',activation_fn=None,scope='layer41',reuse=reuse)
        #print(out.shape,'out')
        out = tf.add(out, avr_raw)
        return out
    
    
    
    
    
def PICNN_V1_3(input,frm_num,reuse=False):  # Unet  use 4 images to test
    with tf.variable_scope("PICNN_V1_3", reuse=reuse) as v1_3:
        n=input
        #print(n[1].shape,'input') 

        #common1=tf.cast([],tf.float32)
        nn1=[None]*frm_num
        common10=[None]*frm_num
        concat1=[None]*frm_num
        for i in range(frm_num):
            nn1[i]=slim.conv2d(n[i],32,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer1_%s'%i,reuse=reuse)
            common10[i]=tf.expand_dims(nn1[i],1,name='common1_%s'%i)  
        common1=tf.concat(common10,axis=1,name='common1')
        #print(common1.shape,'common1')
        pool1=slim.max_pool3d(common1,[6,1,1],[6,1,1],padding='SAME',scope='pool1')
        pool1=tf.squeeze(pool1,[1],name='layer2')
        #print(pool1.shape,'pool1')
        for i in range(frm_num):
            concat1[i]=tf.concat([pool1,nn1[i]],axis=3,name='layer3_%s'%i)
        
        nn2=[None]*frm_num
        nn2_1=[None]*frm_num
        common20=[None]*frm_num    
        concat2=[None]*frm_num
        for i in range(frm_num):
            nn2[i]=slim.conv2d(concat1[i],64,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer4_%s'%i,reuse=reuse)
            nn2_1[i]=slim.conv2d(nn2[i],64,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer5_%s'%i,reuse=reuse)
            common20[i]=tf.expand_dims(nn2_1[i],1)
        common2=tf.concat(common20,axis=1,name='common2')
        #print(common2.shape,'common2')
        pool2=slim.max_pool3d(common2,[6,1,1],[6,1,1],padding='SAME',scope='pool2')
        pool2=tf.squeeze(pool2,[1],name='layer6')
        #print(pool2.shape,'pool2')
        for i in range(frm_num):
            concat2[i]=tf.concat([pool2,nn2_1[i]],axis=3,name='layer7_%s'%i)


        nn3=[None]*frm_num
        nn3_1=[None]*frm_num
        common30=[None]*frm_num    
        concat3=[None]*frm_num
        for i in range(frm_num):
            nn3[i]=slim.conv2d(concat2[i],128,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer8_%s'%i,reuse=reuse) 
            nn3_1[i]=slim.conv2d(nn3[i],128,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer9_%s'%i,reuse=reuse)
            common30=tf.expand_dims(nn3_1[i],1)
        common3=tf.concat(common30,axis=1,name='common3')
        #print(common3.shape,'common3')
        pool3=slim.max_pool3d(common3,[6,1,1],[6,1,1],padding='SAME',scope='pool3')
        pool3=tf.squeeze(pool3,[1],name='layer10')
        #print(pool3.shape,'pool3')
        for i in range(frm_num):
            concat3[i]=tf.concat([pool3,nn3_1[i]],axis=3,name='layer11_%s'%i)
        
        
        
        nn4=[None]*frm_num
        nn4_1=[None]*frm_num
        common40=[None]*frm_num    
        concat4=[None]*frm_num
        for i in range(frm_num):     #conv4
            nn4[i]=slim.conv2d(concat3[i],256,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer12_%s'%i,reuse=reuse)  
            nn4_1[i]=slim.conv2d(nn4[i],256,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer13_%s'%i,reuse=reuse)
            common40[i]=tf.expand_dims(nn4_1[i],1)
        common4=tf.concat(common40,axis=1,name='common4')   
        #print(common4.shape,'common4')
        pool4=slim.max_pool3d(common4,[6,1,1],[6,1,1],padding='SAME',scope='pool4')
        pool4=tf.squeeze(pool4,[1],name='layer14')
        #print(pool4.shape,'pool4')
        for i in range(frm_num):
            concat4[i]=tf.concat([pool4,nn4_1[i]],axis=3,name='layer15_%s'%i) 



        nn5=[None]*frm_num
        nn5_1=[None]*frm_num
        common50=[None]*frm_num
        denn5=[None]*frm_num
        denn5_1=[None]*frm_num
        concat5=[None]*frm_num
        for i in range(frm_num):       #conv5
            nn5[i]=slim.conv2d(concat4[i],384,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer16_%s'%i,reuse=reuse)
            nn5_1[i]=slim.conv2d(nn5[i], 384,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer17_%s'%i,reuse=reuse)
            #print(nn5_1[i].shape,'nn5[i]')
            denn5[i]=slim.conv2d(nn5_1[i], 384,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer18_%s'%i,reuse=reuse)            
            denn5_1[i]=my_deconv(denn5[i],nn5[i],384,384,name='layer19%s'%i) 
            #print(denn5[i].shape,'denn5[i]')
            common50[i]=tf.expand_dims(denn5_1[i],1)
        common5=tf.concat(common50,axis=1,name='common5')
        #print(common5.shape,'common5')
        pool5=slim.max_pool3d(common5,[6,1,1],[6,1,1],padding='SAME',scope='pool5')
        pool5=tf.squeeze(pool5,[1],name='layer20')#256
        for i in range(frm_num):
            concat5[i]=tf.concat([pool5,denn5_1[i],nn5[i]],axis=3,name='layer21_%s'%i) #384*3
            


        back_nn4=[None]*frm_num
        denn4=[None]*frm_num
        back_nn4_1=[None]*frm_num
        decommon40=[None]*frm_num
        concat6=[None]*frm_num
        for i in range(frm_num):     #deconv4
            back_nn4[i]=slim.conv2d(concat5[i],384,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer22_%s'%i,reuse=reuse) 
            back_nn4_1[i]=slim.conv2d(back_nn4[i],384,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer23_%s'%i,reuse=reuse) 
            #print(back_nn4_1[i].shape,'back_nn4[i]')
            denn4[i]=my_deconv(back_nn4_1[i],nn4[i],256,384,name='layer24_%s'%i) 
            #print(denn4[i].shape,'denn4[i]')
            #decommon4=tf.concat([tf.expand_dims(denn4,1),decommon4],axis=1,name='decommon4%s'%i)
            decommon40[i]=tf.expand_dims(denn4[i],1)
        decommon4=tf.concat(decommon40,axis=1,name='decommon4')     
        #print(decommon4.shape,'decommon4')
        depool4=slim.max_pool3d(decommon4,[6,1,1],[6,1,1],padding='SAME',scope='depool4')
        depool4=tf.squeeze(depool4,[1],name='layer25')
        for i in range(frm_num):
            concat6[i]=tf.concat([depool4,denn4[i],nn4[i]],axis=3,name='layer26_%s'%i) #128*3

        back_nn3=[None]*frm_num
        back_nn3_1=[None]*frm_num
        denn3=[None]*frm_num
        decommon30=[None]*frm_num
        concat7=[None]*frm_num
        for i in range(frm_num):     #deconv3
            back_nn3[i]=slim.conv2d(concat6[i],256,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer27_%s'%i,reuse=reuse) 
            back_nn3_1[i]=slim.conv2d(back_nn3[i],256,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.relu,scope='layer28_%s'%i,reuse=reuse)
            denn3[i]=my_deconv(back_nn3_1[i],nn3[i],128,256,name='layer29_%s'%i) 
            #print(back_nn3_1[i].shape,'back_nn3[i]')
            #print(denn3[i].shape,'denn3[i]')
            decommon30[i]=tf.expand_dims(denn3[i],1)
        decommon3=tf.concat(decommon30,axis=1,name='decommon3')
        #print(decommon3.shape,'decommon3')
        depool3=slim.max_pool3d(decommon3,[6,1,1],[6,1,1],padding='SAME',scope='depool3')
        depool3=tf.squeeze(depool3,[1],name='layer30')
        for i in range(frm_num):
            concat7[i]=tf.concat([depool3,denn3[i],nn3[i]],axis=3,name='layer31_%s'%i) #192*3
      
            

        back_nn2=[None]*frm_num
        denn2=[None]*frm_num
        decommon20=[None]*frm_num
        concat8=[None]*frm_num
        for i in range(frm_num):     #deconv2
            back_nn2[i]=slim.conv2d(concat7[i],128,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer32_%s'%i,reuse=reuse) #64
            back_nn2[i]=slim.conv2d(back_nn2[i],128,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer33_%s'%i,reuse=reuse) #64->64
            denn2[i]=my_deconv(back_nn2[i],nn2[i],64,128,name='layer34_%s'%i) 
            #print(back_nn2[i].shape,'back_nn2[i]')
            #print(denn2[i].shape,'denn2[i]')
            decommon20[i]=tf.expand_dims(denn2[i],1)
        decommon2=tf.concat(decommon20,axis=1,name='decommon2')
        #print(decommon2.shape,'decommon2')
        depool2=slim.max_pool3d(decommon2,[6,1,1],[6,1,1],padding='SAME',scope='depool2')
        depool2=tf.squeeze(depool2,[1],name='layer35')
        for i in range(frm_num):
            concat8[i]=tf.concat([depool2,denn2[i],nn2[i]],axis=3,name='layer36_%s'%i) 
               
        back_nn1=[None]*frm_num
        #denn1=[None]*frm_num
        decommon10=[None]*frm_num
        for i in range(frm_num):     #deconv2
            back_nn1[i]=slim.conv2d(concat8[i],64,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer37_%s'%i,reuse=reuse) #64
            back_nn1[i]=slim.conv2d(back_nn1[i],32,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer38_%s'%i,reuse=reuse) #64->64
            decommon10[i]=tf.expand_dims(back_nn1[i],1)
        decommon1=tf.concat(decommon10,axis=1,name='decommon1')
        #print(decommon1.shape,'decommon1')
        depool1=slim.max_pool3d(decommon1,[6,1,1],[6,1,1],padding='SAME',scope='depool1')
        depool1=tf.squeeze(depool1,[1],name='layer39')
        
        
               
        
        
        
        n_final=slim.conv2d(depool1,32,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer40_%s'%i,reuse=reuse) 
        #print(n_final.shape,'n_final')
        out=slim.conv2d(n_final,4,[3,3],stride=1,padding='SAME',activation_fn=None,scope='layer41',reuse=reuse)
        #print(out.shape,'out')
        return out


def PICNN_V1_4(input,frm_num,reuse=False):  # Unet  use 4 images to test
    with tf.variable_scope("PICNN_V1_4", reuse=reuse) as v1_4:
        n=input
        #print(n[1].shape,'input') 

        #common1=tf.cast([],tf.float32)
        nn1=[None]*frm_num
        common10=[None]*frm_num
        concat1=[None]*frm_num
        for i in range(frm_num):
            nn1[i]=slim.conv2d(n[i],32,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer1_%s'%i,reuse=reuse)
            common10[i]=tf.expand_dims(nn1[i],1,name='common1_%s'%i)  
        common1=tf.concat(common10,axis=1,name='common1')
        #print(common1.shape,'common1')
        pool1=slim.max_pool3d(common1,[6,1,1],[6,1,1],padding='SAME',scope='pool1')
        pool1=tf.squeeze(pool1,[1],name='layer2')
        #print(pool1.shape,'pool1')
        for i in range(frm_num):
            concat1[i]=tf.concat([pool1,nn1[i]],axis=3,name='layer3_%s'%i)
        
        nn2=[None]*frm_num
        nn2_1=[None]*frm_num
        common20=[None]*frm_num    
        concat2=[None]*frm_num
        for i in range(frm_num):
            nn2[i]=slim.conv2d(concat1[i],64,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer4_%s'%i,reuse=reuse)
            nn2_1[i]=slim.conv2d(nn2[i],64,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer5_%s'%i,reuse=reuse)
            common20[i]=tf.expand_dims(nn2_1[i],1)
        common2=tf.concat(common20,axis=1,name='common2')
        #print(common2.shape,'common2')
        pool2=slim.max_pool3d(common2,[6,1,1],[6,1,1],padding='SAME',scope='pool2')
        pool2=tf.squeeze(pool2,[1],name='layer6')
        #print(pool2.shape,'pool2')
        for i in range(frm_num):
            concat2[i]=tf.concat([pool2,nn2_1[i]],axis=3,name='layer7_%s'%i)


        nn3=[None]*frm_num
        nn3_1=[None]*frm_num
        common30=[None]*frm_num    
        concat3=[None]*frm_num
        for i in range(frm_num):
            nn3[i]=slim.conv2d(concat2[i],128,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer8_%s'%i,reuse=reuse) 
            nn3_1[i]=slim.conv2d(nn3[i],128,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer9_%s'%i,reuse=reuse)
            common30=tf.expand_dims(nn3_1[i],1)
        common3=tf.concat(common30,axis=1,name='common3')
        #print(common3.shape,'common3')
        pool3=slim.max_pool3d(common3,[6,1,1],[6,1,1],padding='SAME',scope='pool3')
        pool3=tf.squeeze(pool3,[1],name='layer10')
        #print(pool3.shape,'pool3')
        for i in range(frm_num):
            concat3[i]=tf.concat([pool3,nn3_1[i]],axis=3,name='layer11_%s'%i)
        
        
        
        nn4=[None]*frm_num
        nn4_1=[None]*frm_num
        common40=[None]*frm_num    
        concat4=[None]*frm_num
        for i in range(frm_num):     #conv4
            nn4[i]=slim.conv2d(concat3[i],256,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer12_%s'%i,reuse=reuse)  
            nn4_1[i]=slim.conv2d(nn4[i],256,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer13_%s'%i,reuse=reuse)
            common40[i]=tf.expand_dims(nn4_1[i],1)
        common4=tf.concat(common40,axis=1,name='common4')   
        #print(common4.shape,'common4')
        pool4=slim.max_pool3d(common4,[6,1,1],[6,1,1],padding='SAME',scope='pool4')
        pool4=tf.squeeze(pool4,[1],name='layer14')
        #print(pool4.shape,'pool4')
        for i in range(frm_num):
            concat4[i]=tf.concat([pool4,nn4_1[i]],axis=3,name='layer15_%s'%i) 



        nn5=[None]*frm_num
        nn5_1=[None]*frm_num
        common50=[None]*frm_num
        denn5=[None]*frm_num
        denn5_1=[None]*frm_num
        concat5=[None]*frm_num
        for i in range(frm_num):       #conv5
            nn5[i]=slim.conv2d(concat4[i],384,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer16_%s'%i,reuse=reuse)
            nn5_1[i]=slim.conv2d(nn5[i], 384,[4,4],stride=2,padding='SAME',activation_fn=tf.nn.elu,scope='layer17_%s'%i,reuse=reuse)
            #print(nn5_1[i].shape,'nn5[i]')
            denn5[i]=slim.conv2d(nn5_1[i], 384,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer18_%s'%i,reuse=reuse)            
            denn5_1[i]=my_deconv(denn5[i],nn5[i],384,384,name='layer19%s'%i) 
            #print(denn5[i].shape,'denn5[i]')
            common50[i]=tf.expand_dims(denn5_1[i],1)
        common5=tf.concat(common50,axis=1,name='common5')
        #print(common5.shape,'common5')
        pool5=slim.max_pool3d(common5,[6,1,1],[6,1,1],padding='SAME',scope='pool5')
        pool5=tf.squeeze(pool5,[1],name='layer20')#256
        for i in range(frm_num):
            concat5[i]=tf.concat([pool5,denn5_1[i],nn5[i]],axis=3,name='layer21_%s'%i) #384*3
            


        back_nn4=[None]*frm_num
        denn4=[None]*frm_num
        back_nn4_1=[None]*frm_num
        decommon40=[None]*frm_num
        concat6=[None]*frm_num
        for i in range(frm_num):     #deconv4
            back_nn4[i]=slim.conv2d(concat5[i],384,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer22_%s'%i,reuse=reuse) 
            back_nn4_1[i]=slim.conv2d(back_nn4[i],384,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer23_%s'%i,reuse=reuse) 
            #print(back_nn4_1[i].shape,'back_nn4[i]')
            denn4[i]=my_deconv(back_nn4_1[i],nn4[i],256,384,name='layer24_%s'%i) 
            #print(denn4[i].shape,'denn4[i]')
            #decommon4=tf.concat([tf.expand_dims(denn4,1),decommon4],axis=1,name='decommon4%s'%i)
            decommon40[i]=tf.expand_dims(denn4[i],1)
        decommon4=tf.concat(decommon40,axis=1,name='decommon4')     
        #print(decommon4.shape,'decommon4')
        depool4=slim.max_pool3d(decommon4,[6,1,1],[6,1,1],padding='SAME',scope='depool4')
        depool4=tf.squeeze(depool4,[1],name='layer25')
        for i in range(frm_num):
            concat6[i]=tf.concat([depool4,denn4[i],nn4[i]],axis=3,name='layer26_%s'%i) #128*3

        back_nn3=[None]*frm_num
        back_nn3_1=[None]*frm_num
        denn3=[None]*frm_num
        decommon30=[None]*frm_num
        concat7=[None]*frm_num
        for i in range(frm_num):     #deconv3
            back_nn3[i]=slim.conv2d(concat6[i],256,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer27_%s'%i,reuse=reuse) 
            back_nn3_1[i]=slim.conv2d(back_nn3[i],256,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.relu,scope='layer28_%s'%i,reuse=reuse)
            denn3[i]=my_deconv(back_nn3_1[i],nn3[i],192,256,name='layer29_%s'%i) 
            #print(back_nn3_1[i].shape,'back_nn3[i]')
            #print(denn3[i].shape,'denn3[i]')
            decommon30[i]=tf.expand_dims(denn3[i],1)
        decommon3=tf.concat(decommon30,axis=1,name='decommon3')
        #print(decommon3.shape,'decommon3')
        depool3=slim.max_pool3d(decommon3,[6,1,1],[6,1,1],padding='SAME',scope='depool3')
        depool3=tf.squeeze(depool3,[1],name='layer30')
        for i in range(frm_num):
            concat7[i]=tf.concat([depool3,denn3[i],nn3[i]],axis=3,name='layer31_%s'%i) #192*3
      
            

        back_nn2=[None]*frm_num
        denn2=[None]*frm_num
        decommon20=[None]*frm_num
        concat8=[None]*frm_num
        for i in range(frm_num):     #deconv2
            back_nn2[i]=slim.conv2d(concat7[i],192,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer32_%s'%i,reuse=reuse) #64
            back_nn2[i]=slim.conv2d(back_nn2[i],192,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer33_%s'%i,reuse=reuse) #64->64
            denn2[i]=my_deconv(back_nn2[i],nn2[i],64,192,name='layer34_%s'%i) 
            #print(back_nn2[i].shape,'back_nn2[i]')
            #print(denn2[i].shape,'denn2[i]')
            decommon20[i]=tf.expand_dims(denn2[i],1)
        decommon2=tf.concat(decommon20,axis=1,name='decommon2')
        #print(decommon2.shape,'decommon2')
        depool2=slim.max_pool3d(decommon2,[6,1,1],[6,1,1],padding='SAME',scope='depool2')
        depool2=tf.squeeze(depool2,[1],name='layer35')
        for i in range(frm_num):
            concat8[i]=tf.concat([depool2,denn2[i],nn2[i]],axis=3,name='layer36_%s'%i) 
               
        back_nn1=[None]*frm_num
        #denn1=[None]*frm_num
        decommon10=[None]*frm_num
        for i in range(frm_num):     #deconv2
            back_nn1[i]=slim.conv2d(concat8[i],64,[1,1],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer37_%s'%i,reuse=reuse) #64
            back_nn1[i]=slim.conv2d(back_nn1[i],32,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer38_%s'%i,reuse=reuse) #64->64
            decommon10[i]=tf.expand_dims(back_nn1[i],1)
        decommon1=tf.concat(decommon10,axis=1,name='decommon1')
        #print(decommon1.shape,'decommon1')
        depool1=slim.max_pool3d(decommon1,[6,1,1],[6,1,1],padding='SAME',scope='depool1')
        depool1=tf.squeeze(depool1,[1],name='layer39')
        
        
               
        
        
        
        n_final=slim.conv2d(depool1,32,[3,3],stride=1,padding='SAME',activation_fn=tf.nn.elu,scope='layer40',reuse=reuse) 
        #print(n_final.shape,'n_final')
        out=slim.conv2d(n_final,4,[3,3],stride=1,padding='SAME',activation_fn=None,scope='layer41',reuse=reuse)
        #print(out.shape,'out')
        return out