import numpy as np
from math import ceil
import matplotlib.pyplot as plt
#from motion_blur.generate_trajectory import Trajectory
from generate_trajectory import Trajectory
import cv2,os
import tensorflow as tf

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def gen_example_v0(psf_list, shape=(32, 32), psf_num=1):#32 32#16*16
    tfrecords_features = {}
    tfrecords_features['psf'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=psf_list))
    tfrecords_features['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(shape)))
    tfrecords_features['psf_num'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[psf_num]))
    return tf.train.Example(features=tf.train.Features(feature=tfrecords_features))
        
def gen_example_v1(psf_list, shape=(16, 16), psf_num=1):#32 32#16*16
    tfrecords_features = {}
    tfrecords_features['psf'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=psf_list))
    tfrecords_features['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(shape)))
    tfrecords_features['psf_num'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[psf_num]))
    return tf.train.Example(features=tf.train.Features(feature=tfrecords_features))
        
class PSF(object):
    def __init__(self, canvas=None, trajectory=None, fraction=None, path_to_save=None):
        if canvas is None:
            self.canvas = (canvas, canvas)
        else:
            self.canvas = (canvas, canvas)
        if trajectory is None:
            self.trajectory = Trajectory(canvas=canvas, expl=0.005).fit(show=False, save=False)
        else:
            self.trajectory = trajectory.x
        if fraction is None:
            self.fraction = [1]#[1/100, 1/20, 1/10, 1/5, 1/2, 1/1.5, 1]
        else:
            self.fraction = fraction
        self.path_to_save = path_to_save
        self.PSFnumber = len(self.fraction)#the original code
        #print(self.PSFnumber)
        self.iters = len(self.trajectory)
        self.PSFs = []

    def fit(self, show=False, save=False):
        PSF = np.zeros(self.canvas)

        triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
        triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
        for j in range(self.PSFnumber):
            if j == 0:
                prevT = 0
            else:
                prevT = self.fraction[j - 1]

            for t in range(len(self.trajectory)):
                # print(j, t)
                if (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t - 1):
                    t_proportion = 1
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t - 1):
                    t_proportion = self.fraction[j] * self.iters - (t - 1)
                elif (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t):
                    t_proportion = t - (prevT * self.iters)
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t):
                    t_proportion = (self.fraction[j] - prevT) * self.iters
                else:
                    t_proportion = 0

                m2 = int(np.minimum(self.canvas[1] - 2, np.maximum(1, np.math.floor(self.trajectory[t].real))))
                M2 = int(m2 + 1)
                m1 = int(np.minimum(self.canvas[0] - 2, np.maximum(1, np.math.floor(self.trajectory[t].imag))))
                M1 = int(m1 + 1)

                PSF[m1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - m1
                )
                PSF[m1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - m1
                )
                PSF[M1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - M1
                )
                PSF[M1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - M1
                )

            self.PSFs.append(PSF / (self.iters))
        if show or save:
            #print(self.PSFs[1][20:30,20:30])
            self.__plot_canvas(show, save)

        return self.PSFs

    def __plot_canvas(self, show, save):
        if len(self.PSFs) == 0:
            raise Exception("Please run fit() method first.")
        else:
            plt.close()
            fig, axes = plt.subplots(1, self.PSFnumber, figsize=(10, 10))
            for i in range(self.PSFnumber):
                axes.imshow(self.PSFs[i], cmap='gray')
            if show and save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                plt.savefig(self.path_to_save)
                plt.show()
            elif save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                plt.savefig(self.path_to_save)
            elif show:
                plt.show()




if __name__ == '__main__':
    #params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
    #trajectory = Trajectory(canvas=64, max_len=5, expl=np.random.choice(params)).fit()
    #psf = PSF(canvas=64, trajectory=trajectory,path_to_save="/DATA1/dlnr_zcz/psf/").fit(show=True,save=True)
    
    path_to_save = "/DATA1/dlnr_zcz/psf_32(1)/"
    create_dir(path_to_save)
    number = 1000000
    example_num=50000
    record_start_cnt=0
    record_num = number // example_num
    if record_num * example_num != number:
        record_num += 1

    for i in range(record_num):
        tfrecord_name = path_to_save + "N%05d.tfrecord" % (i + 1 + record_start_cnt)
        writer = tf.python_io.TFRecordWriter(tfrecord_name)
        for j in range(min(number - example_num * i, example_num)):
            params = [0.03, 0.009, 0.008, 0.007, 0.005, 0.004]
            max_len = np.random.choice(np.arange(1,31),p=[0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06])
            trajectory = Trajectory(canvas=32, max_len=max_len, expl=np.random.choice(params)).fit()#60#np.random.randint(3,15)
            psf = PSF(canvas=32, trajectory=trajectory,path_to_save=path_to_save).fit(show=False,save=False)
            #print(np.sum(psf[0]))
            psfchoice=0#np.random.choice(np.arange(7))
            cv2.normalize(psf[psfchoice], psf[psfchoice],1.0,0.0,cv2.NORM_L1)#, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)#NORM_MINMAX
            #print(psf[2].shape)
            #print(psf[0])
            #print(np.sum(psf[psfchoice]))
            print("process %d in %d"%(example_num * i+j, number))
            psf_bytes = (psf[psfchoice]).tobytes()
            example = gen_example_v0([psf_bytes], shape=(32, 32), psf_num=1)
            example_serial = example.SerializeToString()
            writer.write(example_serial)               
        writer.close()      

'''        
    for i in range(record_num):
        tfrecord_name = path_to_save + "N%05d.tfrecord" % (i + 1 + record_start_cnt)
        writer = tf.python_io.TFRecordWriter(tfrecord_name)
        for j in range(min(number - example_num * i, example_num)):
            params = [0.03, 0.009, 0.008, 0.007, 0.005, 0.004]
            max_len = np.random.choice(np.arange(1,16),p=[0.02,0.02,0.02,0.02,0.02,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09])
            trajectory = Trajectory(canvas=16, max_len=max_len, expl=np.random.choice(params)).fit()#60#np.random.randint(3,15)
            psf = PSF(canvas=16, trajectory=trajectory,path_to_save=path_to_save).fit(show=False,save=False)
            #print(np.sum(psf[0]))
            psfchoice=0#np.random.choice(np.arange(7))
            cv2.normalize(psf[psfchoice], psf[psfchoice],1.0,0.0,cv2.NORM_L1)#, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)#NORM_MINMAX
            #print(psf[2].shape)
            #print(psf[0])
            #print(np.sum(psf[psfchoice]))
            print("process %d in %d"%(example_num * i+j, number))
            psf_bytes = (psf[psfchoice]).tobytes()
            example = gen_example_v1([psf_bytes], shape=(16, 16), psf_num=1)
            example_serial = example.SerializeToString()
            writer.write(example_serial)               
        writer.close()        
'''    
        
"""
        pad=np.zeros(psf[0].shape)
        psf_1=np.stack([psf[0],pad,pad,pad],axis=-1)
        psf_2=np.stack([pad,psf[0],pad,pad],axis=-1)
        psf_3=np.stack([pad,pad,psf[0],pad],axis=-1)
        psf_4=np.stack([pad,pad,pad,psf[0]],axis=-1)
        psf_final=np.stack([psf_1,psf_2,psf_3,psf_4],axis=-1)
        #print(psf[0].shape)#64*64
        #np.savetxt(path_to_save+'%d.txt'%(i+1),psf[0])
        #cv2.imwrite(path_to_save+'N%5d.png' % (i+1),(psf[0]*20*255).astype(np.uint8))
        psf=np.array(psf_final)
        print(np.sum(psf_final))
        #print(psf.ndim)
        print(psf_final.shape)
  '''
    #print(np.loadtxt(path_to_save+'1.txt')[35:40,25:30])
        #print(psf[0][35:40,25:30],'b')
    #psf = PSF(canvas=128, path_to_save="/DATA1/dlnr_zcz/psf/")
    #psf.fit(show=True, save=False)
"""
