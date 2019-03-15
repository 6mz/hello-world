# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:26:12 2019

@author: zWX618024
"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil


class Trajectory(object):
    def __init__(self, canvas=128, iters=1000, max_len=60, expl=None, path_to_save=None):
        """
        Generates a variety of random motion trajectories in continuous domain as in [Boracchi and Foi 2012]. Each
        trajectory consists of a complex-valued vector determining the discrete positions of a particle following a
        2-D random motion in continuous domain. The particle has an initial velocity vector which, at each iteration,
        is affected by a Gaussian perturbation and by a deterministic inertial component, directed toward the
        previous particle position. In addition, with a small probability, an impulsive (abrupt) perturbation aiming
        at inverting the particle velocity may arises, mimicking a sudden movement that occurs when the user presses
        the camera button or tries to compensate the camera shake. At each step, the velocity is normalized to
        guarantee that trajectories corresponding to equal exposures have the same length. Each perturbation (
        Gaussian, inertial, and impulsive) is ruled by its own parameter. Rectilinear Blur as in [Boracchi and Foi
        2011] can be obtained by setting anxiety to 0 (when no impulsive changes occurs
        :param canvas: size of domain where our trajectory os defined.
        :param iters: number of iterations for definition of our trajectory.
        :param max_len: maximum length of our trajectory.
        :param expl: this param helps to define probability of big shake. Recommended expl = 0.005.
        :param path_to_save: where to save if you need.
        """
        self.canvas = canvas
        self.iters = iters
        self.max_len = max_len
        if expl is None:
            self.expl = 0.1 * np.random.uniform(0, 1)
        else:
            self.expl = expl
        if path_to_save is None:
            pass
        else:
            self.path_to_save = path_to_save
        self.tot_length = None
        self.big_expl_count = None
        self.x = None
        self.y = None
        self.z = None
        self.w = None


    def fit(self, show=False, save=False):
        """
        Generate motion, you can save or plot, coordinates of motion you can find in x property.
        Also you can fin properties tot_length, big_expl_count.
        :param show: default False.
        :param save: default False.
        :return: x (vector of motion).
        """
        tot_length = 0
        big_expl_count = 0                
        
        tot_length2 = 0
        big_expl_count2 = 0
        
        tot_length3 = 0
        big_expl_count3 = 0
        
        tot_length4 = 0
        big_expl_count4 = 0
        
        # how to be near the previous position
        # TODO: I can change this paramether for 0.1 and make kernel at all image
        centripetal = 0.7 * np.random.uniform(0, 1)
        # probability of big shake
        prob_big_shake = 0.2 * np.random.uniform(0, 1)#original is 0.2
        # term determining, at each sample, the random component of the new direction
        gaussian_shake = 10 * np.random.uniform(0, 1)
        init_angle1 = 360 * np.random.uniform(0, 1)
        init_angle2 = 360 * np.random.uniform(0, 1)
        init_angle3 = 360 * np.random.uniform(0, 1)
        init_angle4 = 360 * np.random.uniform(0, 1)
       
        
        
        

        img_v0 = np.sin(np.deg2rad(init_angle1))
        real_v0 = np.cos(np.deg2rad(init_angle1))
        
        img_v02 = np.sin(np.deg2rad(init_angle2))
        real_v02 = np.cos(np.deg2rad(init_angle2))
        
        img_v03 = np.sin(np.deg2rad(init_angle3))
        real_v03 = np.cos(np.deg2rad(init_angle3))
        
        img_v04 = np.sin(np.deg2rad(init_angle4))
        real_v04 = np.cos(np.deg2rad(init_angle4))


        v0 = complex(real=real_v0, imag=img_v0)
        v = v0 * self.max_len / (self.iters - 1)

        v02 = complex(real=real_v02, imag=img_v02)
        v2 = v02 * self.max_len / (self.iters - 1)

        v03 = complex(real=real_v03, imag=img_v03)
        v3 = v03 * self.max_len / (self.iters - 1)

        v04 = complex(real=real_v04, imag=img_v04)
        v4 = v04 * self.max_len / (self.iters - 1)

        if self.expl > 0:
            v = v0 * self.expl
            v2 = v02 * self.expl
            v3 = v03 * self.expl
            v4 = v04 * self.expl


        x = np.array([complex(real=0, imag=0)] * (self.iters))
        y = np.array([complex(real=0, imag=60)] * (self.iters))
        z = np.array([complex(real=60, imag=0)] * (self.iters))
        w = np.array([complex(real=60, imag=60)] * (self.iters))


        for t in range(0, self.iters - 1):
            if np.random.uniform() < prob_big_shake * self.expl:
                next_direction = 2 * v * (np.exp(complex(real=0, imag=np.pi + (np.random.uniform() - 0.5))))
                big_expl_count += 1
            else:
                next_direction = 0

            dv = next_direction + self.expl * (
                gaussian_shake * complex(real=np.random.randn(), imag=np.random.randn()) - centripetal * x[t]) * (
                                      self.max_len / (self.iters - 1))

            v += dv
            v = (v / float(np.abs(v))) * (self.max_len / float((self.iters - 1)))
            x[t + 1] = x[t] + v
            tot_length = tot_length + abs(x[t + 1] - x[t])

        # centere the motion
        x += complex(real=-np.min(x.real), imag=-np.min(x.imag))
        x = x - complex(real=x[0].real % 1., imag=x[0].imag % 1.) + complex(1, 1)
        x += complex(real=ceil((self.canvas - max(x.real)) / 2), imag=ceil((self.canvas - max(x.imag)) / 2))

        self.tot_length = tot_length
        self.big_expl_count = big_expl_count
        self.x = x
        print(x)


        for t in range(0, self.iters - 1):
            if np.random.uniform() < prob_big_shake * self.expl:
                next_direction = 2 * v2 * (np.exp(complex(real=0, imag=np.pi + (np.random.uniform() - 0.5))))
                big_expl_count2 += 1
            else:
                next_direction = 0

            dv2 = next_direction + self.expl * (
                gaussian_shake * complex(real=np.random.randn(), imag=np.random.randn()) - centripetal * y[t]) * (
                                      self.max_len / (self.iters - 1))

            v2 += dv2
            v2 = (v2 / float(np.abs(v2))) * (self.max_len / float((self.iters - 1)))
            y[t + 1] = y[t] + v2
            tot_length2 = tot_length2 + abs(y[t + 1] - y[t])

        # centere the motion
        y += complex(real=-np.min(y.real), imag=-np.min(y.imag))
        y = y - complex(real=y[0].real % 1., imag=y[0].imag % 1.) + complex(1, 1)
        y += complex(real=ceil((self.canvas - max(y.real)) / 2), imag=ceil((self.canvas - max(y.imag)) / 2))

        self.tot_length2 = tot_length2
        self.big_expl_count2 = big_expl_count2
        self.y = y
        #print(y)




        for t in range(0, self.iters - 1):
            if np.random.uniform() < prob_big_shake * self.expl:
                next_direction = 2 * v3 * (np.exp(complex(real=0, imag=np.pi + (np.random.uniform() - 0.5))))
                big_expl_count3 += 1
            else:
                next_direction = 0

            dv3 = next_direction + self.expl * (
                gaussian_shake * complex(real=np.random.randn(), imag=np.random.randn()) - centripetal * z[t]) * (
                                      self.max_len / (self.iters - 1))

            v3 += dv3
            v3 = (v3 / float(np.abs(v3))) * (self.max_len / float((self.iters - 1)))
            z[t + 1] = z[t] + v3
            tot_length3 = tot_length3 + abs(z[t + 1] - z[t])

        # centere the motion
        z += complex(real=-np.min(z.real), imag=-np.min(z.imag))
        z = z - complex(real=z[0].real % 1., imag=z[0].imag % 1.) + complex(1, 1)
        z += complex(real=ceil((self.canvas - max(z.real)) / 2), imag=ceil((self.canvas - max(z.imag)) / 2))

        self.tot_length3 = tot_length3
        self.big_expl_count3 = big_expl_count3
        self.z = z



        for t in range(0, self.iters - 1):
            if np.random.uniform() < prob_big_shake * self.expl:
                next_direction = 2 * v4 * (np.exp(complex(real=0, imag=np.pi + (np.random.uniform() - 0.5))))
                big_expl_count4 += 1
            else:
                next_direction = 0

            dv4 = next_direction + self.expl * (
                gaussian_shake * complex(real=np.random.randn(), imag=np.random.randn()) - centripetal * w[t]) * (
                                      self.max_len / (self.iters - 1))

            v4 += dv4
            v4 = (v4 / float(np.abs(v4))) * (self.max_len / float((self.iters - 1)))
            w[t + 1] = w[t] + v4
            tot_length4 = tot_length4 + abs(w[t + 1] - w[t])

        # centere the motion
        w += complex(real=-np.min(w.real), imag=-np.min(w.imag))
        w = w - complex(real=w[0].real % 1., imag=w[0].imag % 1.) + complex(1, 1)
        w += complex(real=ceil((self.canvas - max(w.real)) / 2), imag=ceil((self.canvas - max(w.imag)) / 2))
        #print(w)
        self.tot_length4 = tot_length4
        self.big_expl_count4 = big_expl_count4
        self.w = w


        if show or save:
            self.__plot_canvas(show, save)
        return self

    def __plot_canvas(self, show, save):
        if self.x is None:
            raise Exception("Please run fit() method first")
        else:
            plt.close()
            plt.plot(self.x.real, self.x.imag, '-', color='blue')
            plt.plot(self.y.real, self.y.imag, '-', color='red')
            plt.plot(self.z.real, self.z.imag, '-', color='green')                        
            plt.plot(self.w.real, self.w.imag, '-', color='orange')

            
            plt.xlim((-60, self.canvas))
            plt.ylim((-60, self.canvas))
            if show and save:
                plt.savefig(self.path_to_save)
                plt.show()
            elif save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                plt.savefig(self.path_to_save)
            elif show:
                plt.show()


if __name__ == '__main__':
    trajectory = Trajectory(expl=0.005,
                            path_to_save="/DATA1/dlnr_zcz/trajectory/")
    trajectory.fit(True, False)
    
    
    '''
    if __name__ == '__main__':
    path_to_save = "/DATA1/dlnr_zcz/big_shake_psf/"
    create_dir(path_to_save)
    number = 200000
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
            trajectory = Trajectory(canvas=64, max_len=60, expl=np.random.choice(params)).fit()
            psf = PSF(canvas=64, trajectory=trajectory,path_to_save=path_to_save).fit(show=False,save=True)
            #print(np.sum(psf[0]))
            #cv2.normalize(psf[0], psf[0], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            #print(psf[0])
            #print(np.sum(psf[0]))
            print("process %d in %d"%(example_num * i+j, number))
            psf_bytes = psf[0].tobytes()
            example = gen_example([psf_bytes], shape=(64, 64), psf_num=1)
            example_serial = example.SerializeToString()
            writer.write(example_serial)               
        writer.close()
        '''
