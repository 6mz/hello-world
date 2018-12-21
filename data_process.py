# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:47:52 2018

@author: zWX618024
"""
import numpy as np
#from math import ceil
#import matplotlib.pyplot as plt
import cv2,os
import glob,time

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def gen_gamma(src_rt_files, blur_path, sharp_path):
    create_dir(blur_path)
    create_dir(sharp_path)
    iter_num = 0 
    for src_pa in src_rt_files:
        start_time = time.time()
        src_files = glob.glob(src_pa+"*.png")
        src_num = len(src_files)
        print(src_num)
        pic_start_path = sorted(src_files)[0]
        pic_path,pic_name = os.path.split(pic_start_path)
        print(pic_path,pic_name)
        pic_number,pic_class = os.path.splitext(pic_name)
        pic_number = int(pic_number)
        frm_num = np.random.choice(np.arange(7,13),1)[0]
        frm_total = frm_num 
        while(frm_total<=src_num):
            print(frm_num)
            iter_start = frm_total + pic_number -frm_num
            img_sum = np.float32(cv2.imread(src_pa+"%06d.png" % (iter_start)))
            for i in range(1,frm_num):
                img1 = np.float32(cv2.imread(src_pa+"%06d.png" % (iter_start+i)))
                img_sum = img_sum + img1
            img_sum = img_sum/frm_num
            img_sum = np.clip(img_sum,0,255)
            img_sharp = cv2.imread(src_pa+"%06d.png" % (iter_start+ (1+frm_num)//2))
            iter_num = iter_num + 1
            cv2.imwrite(blur_path+"B%06d.png" % iter_num,img_sum)
            cv2.imwrite(blur_path+"S%06d.png" % iter_num,img_sharp)
            print("iter: %d, time: %4.4fs " %(iter_num, time.time()-start_time))
            frm_num = np.random.choice(np.arange(7,13),1)[0]
            frm_total = frm_total + frm_num

def gen_lin(src_rt_files, blur_path, sharp_path):
    create_dir(blur_path)
    create_dir(sharp_path)
    gamma = 2.2
    iter_num = 0 
    for src_pa in src_rt_files:
        start_time = time.time()
        src_files = glob.glob(src_pa+"*.png")
        src_num = len(src_files)
        print(src_num)
        pic_start_path = sorted(src_files)[0]
        pic_path,pic_name = os.path.split(pic_start_path)
        print(pic_path,pic_name)
        pic_number,pic_class = os.path.splitext(pic_name)
        pic_number = int(pic_number)
        frm_num = np.random.choice(np.arange(7,13),1)
        frm_total = frm_num 
        while(frm_total<=src_num):
            iter_start = frm_total + pic_number -frm_num
            img_sum = np.float32(cv2.imread(src_pa+"%06d.png" % (iter_start)))
            img_sum = pow(img_sum,gamma)
            for i in range(1,frm_num):
                img1 = np.float32(cv2.imread(src_pa+"%06d.png" % (iter_start+i)))
                img1 = pow(img1,gamma)
                img_sum = img_sum + img1
            img_sum = img_sum/frm_num
            img_sum = img_sum/pow(255,gamma-1)
            img_sum = np.clip(img_sum,0,255)
            img_sharp = np.float32(cv2.imread(src_pa+"%06d.png" % (iter_start+ (1+frm_num)//2)))
            img_sharp = pow(img_sharp,gamma)/(pow(255,gamma-1))
            img_sharp = np.clip(img_sharp,0,255)
            iter_num = iter_num + 1
            cv2.imwrite(blur_path+"B%06d.png" % iter_num,img_sum)
            cv2.imwrite(sharp_path+"S%06d.png" % iter_num,img_sharp)
            print("iter: %d, time: %4.4fs " %(iter_num, time.time()-start_time))
            frm_num = np.random.choice(np.arange(7,13),1)
            frm_total = frm_total + frm_num
   


if __name__ == '__main__':
    
    src_path = "/DATA1/dlnr_zcz/train/"
    blur_path = "/DATA1/dlnr_zcz/train/blur_gamma/"
    sharp_path = "/DATA1/dlnr_zcz/train/sharp_gamma/"   
    src_rt_files = glob.glob(src_path+'*/')
    gen_gamma(src_rt_files,blur_path,sharp_path)