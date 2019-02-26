from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.image import transform
import numpy as np
import glob
from skimage import filters
import math,cv2
import tensorlayer as tl
from model_zcz import *

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('tmp_res_dir', './picnn_blur_denoise/tmp/8_/', 'the tmp result during training ')

#ps = 256  # patch size for training  512
frm_num=4 #burst frames
m_num = 6
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_example(example_proto):
    dics = {}
    # dics['label'] = tf.FixedLenFeature(shape=[],dtype=tf.int64)
    dics['clean_img'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
    dics['shape'] = tf.FixedLenFeature(shape=[2], dtype=tf.int64)
    dics['img_num'] = tf.FixedLenFeature(shape=[1], dtype=tf.int64)
    parsed_example = tf.parse_single_example(serialized=example_proto,features=dics)

    clean_img = parsed_example['clean_img']
    clean_img = tf.decode_raw(clean_img, tf.uint16)

    clean_img = tf.reshape(clean_img, [300, 300, 4])
    #shape = parsed_example['shape']
    #img_num = parsed_example['img_num']

    return clean_img



def parse_h(h_proto):
    dics = {}
    # dics['label'] = tf.FixedLenFeature(shape=[],dtype=tf.int64)
    dics['Homo'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
    dics['shape'] = tf.FixedLenFeature(shape=[2], dtype=tf.int64)
    dics['m_num'] = tf.FixedLenFeature(shape=[1], dtype=tf.int64)
    parsed_h = tf.parse_single_example(serialized=h_proto,features=dics)

    homo = parsed_h['Homo']
    homo = tf.decode_raw(homo, tf.float32)
    
    #shape = parsed_h['shape']
    #m_num = parsed_h['m_num']
    
    #print(m_num)
    homo = tf.reshape(homo,[9,8])

    return homo#, shape,m_num



def parse_psf(psf_proto):
    dics = {}
    # dics['label'] = tf.FixedLenFeature(shape=[],dtype=tf.int64)
    dics['psf'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
    dics['shape'] = tf.FixedLenFeature(shape=[2], dtype=tf.int64)
    dics['psf_num'] = tf.FixedLenFeature(shape=[1], dtype=tf.int64)
    parsed_psf = tf.parse_single_example(serialized=psf_proto,features=dics)

    psf = parsed_psf['psf']
    psf = tf.decode_raw(psf, tf.float64)

    psf = tf.reshape(psf, [64,64])
    #shape = parsed_example['shape']
    #img_num = parsed_example['img_num']

    return psf


def noise_profile_v2(iso):
    paralist=[(0.00000009, 0.00130705, 0.00026902, -0.20318409), (0.00000013, 0.00013817, 0.00024098, 0.01151728)]
    if iso > 5000:
        index = 0
    else:
        index = 1
    (B_A, B_B, K_A, K_B) = paralist[index]

    #affine variance model
    b = (-(K_A * iso + K_B)* 16.0 + (B_A * math.pow(iso, 2) + B_B * iso)) / (255.0 * 255.0)
    a = (K_A * iso + K_B) / 255.0
    return a,b

def noise_profile_v2_voga(iso):
    paralist=[(0.00000020,0.00015192,0.00023078,0.00595555)]
    index = 0
    (B_A, B_B, K_A, K_B) = paralist[index]
    
    #affine variance model
    b = (-(K_A * iso + K_B)* 16.0 + (B_A * math.pow(iso, 2) + B_B * iso)) / (255.0 * 255.0)
    a = (K_A * iso + K_B) / 255.0
    return a,b

#run vst
def vst_f(a,b,raw):
    #vst
    sigma = math.sqrt(max(b, 0.0))
    g = max(-b, 0.0)/a
    t0 = a*a * 3.0 / 8.0 + sigma*sigma - a*g
    dst_raw = 2.0*tf.sqrt(tf.maximum(raw*a+t0,0))/a

    return dst_raw

def inv_vst_f_n(a,b,raw):
    sigma = math.sqrt(max(b, 0))
    g = max(-b, 0) / a
    sqrt3_2 = math.sqrt(3.0/2.0)
    raw2 = raw * raw
    raw3 = raw2*raw
    dst_raw = raw2 / 4.0 + sqrt3_2 / (raw * 4.0) - 11.0 / (8 * raw2) + 5.0 * sqrt3_2 / (8.0 * raw3) - 1.0 / 8.0 - sigma * sigma / (a * a)
    dst_raw2 = np.maximum(dst_raw, 0) * a + g

    return dst_raw2

def blur_0(img,homo,sh_size,bpg):
    img_1 = tf.image.resize_image_with_crop_or_pad(img,220,220)
    #img_sum = tf.zeros(tf.shape(img_1), tf.float32)
    zeros = tf.zeros(tf.shape(img_1), tf.float32)
    ones = tf.ones(tf.shape(img_1), tf.float32)
    
    #trans =tf.reshape(homo[:,0,:],[bpg,8])
    #img_0 = transform(img,trans,interpolation='BILINEAR')
    #img_11 = tf.image.resize_image_with_crop_or_pad(img_0,240,240)#un-registered
    img_0 = img#after-registration
    img_11 = img_1
    img_sum = img_11

    #print(homo,'homo 109!')
    for i in range(sh_size):
        trans =tf.reshape(homo[:,i,:],[bpg,8])
        #print("112!")
        img_tmp = transform(img_0,trans,interpolation='BILINEAR')
        img_0 = img_tmp
        img_tmp = tf.image.resize_image_with_crop_or_pad(img_tmp,220,220)
        #img_1 = img_tmp
        img_sum = img_tmp+img_sum
    img_sum  = img_sum / (sh_size+1)
    blur_img = tf.where(tf.greater(img_sum, zeros), img_sum, zeros)
    blur_img = tf.where(tf.greater(img_sum, ones), ones, img_sum)
    #print("120!")
    return blur_img

def save_img(raw, dir, name):
    t_raw = np.clip(raw, 0, 1)
    t_raw = t_raw * 255 * 2
    t_raw = np.clip(t_raw, 0, 255).astype(np.uint8)
    for j in range(4):
        cv2.imwrite(dir + '/%s_%d.png' % (name, j), t_raw[:, :, j])


def save_crop_CE_raw(input, path, size):
    #size = (size[0]+2, size[1]+2)
    row_num = size[0]
    column_num = size[1]
    rst_raw = np.ndarray(shape=size, dtype=np.float32)
    rst_raw[1:row_num:2, 1:column_num:2] = input[ :, :, 0]
    rst_raw[1:row_num:2, 0:column_num:2] = input[ :, :, 1]
    rst_raw[0:row_num:2, 1:column_num:2] = input[ :, :, 2]
    rst_raw[0:row_num:2, 0:column_num:2] = input[ :, :, 3]
    #rst_raw[1:row_num:2, 1:column_num:2] = input[:, :, 3]
    #rst_raw[1:row_num:2, 0:column_num:2] = input[:, :, 2]
    #rst_raw[0:row_num:2, 1:column_num:2] = input[:, :, 1]
    #rst_raw[0:row_num:2, 0:column_num:2] = input[:, :, 0]

    #rst_raw = rst_raw[1:-1,1:-1]
    rst_raw = rst_raw * 4095

    rst_raw = np.where(rst_raw > 4095, 4095, rst_raw)
    rst_raw = rst_raw.astype(np.uint16)
    rst_raw.tofile(path)
    return 0

# save the tmp image during training
def save_tmp_res(tmp_in, tmp_gt, tmp_res, dirr):
    a, b = noise_profile_v2(25600)
    tmp_in = inv_vst_f_n(a,b,tmp_in)
    tmp_gt = inv_vst_f_n(a,b,tmp_gt)
    tmp_res = inv_vst_f_n(a,b,tmp_res)
    frame_num = frm_num
    size = (440, 440)
    for jj in range(frame_num):
        raw_tmp = tmp_in[:, :, 4*jj:4*jj+4]
        save_img(raw_tmp, dirr, 'noisy_%d' % jj)
        dst_raw_name = dirr+'/noisy_%d.raw'%jj
        #print(dst_raw_name)
        save_crop_CE_raw(raw_tmp,dst_raw_name,size)
    save_img(tmp_gt, dirr, 'gt')
    dst_raw_name = dirr+'/gt.raw'
    save_crop_CE_raw(tmp_gt,dst_raw_name,size)
    save_img(tmp_res, dirr, 'net_out')

def preprocess_0(clean_img, bit_depth, iso, frm_num,homo,m_num,bpg):
    # a, b = new_noise_profile(iso)
    # max_v = new_cal_max_v(a, b)

    # iso_index = tf.random_uniform(shape=[1], minval=0, maxval=2, dtype=tf.int32)
    # a, b = noise_profile_v2(tf.convert_to_tensor(iso_list)[iso_index])
    a, b = noise_profile_v2_voga(iso)
    #clean_img = tf.random_crop(clean_img, [tf.shape(clean_img)[0], 256, 256, 4])
    
    clean_img = tf.cond(tf.greater(tf.random_normal([1])[0], 0), lambda: tf.transpose(clean_img,[0,2,1,3]), lambda: clean_img)
    clean_img = tf.cond(tf.greater(tf.random_normal([1])[0], 0), lambda: tf.reverse(clean_img,axis=[1]), lambda: clean_img) #verticle flipping
    clean_img = tf.cond(tf.greater(tf.random_normal([1])[0], 0), lambda: tf.reverse(clean_img,axis=[2]), lambda: clean_img)#herizontal flipping
    
    #clean_img = tf.image.random_flip_left_right(clean_img)
    #clean_img = tf.image.random_flip_up_down(clean_img)
    #single_homo_size = homo_size[0]/frm_num
    clean_img = tf.cast(clean_img, tf.float32)
    homo = tf.cast(homo,tf.float32)
    # blc = pow(2, bit_depth - 4)
    # clean_img = clean_img-blc
    zeros = tf.zeros(tf.shape(clean_img), tf.float32)
    clean_img = tf.where(tf.greater(clean_img, zeros), clean_img, zeros)
        
        # input_patch = np.transpose(input_patch, (0,2,1,3))            clean_img=tf.reverse(clean_img,axis=[1])
    clean_img = clean_img / float(pow(2, bit_depth) - 1)
    clean_img_1 = tf.image.resize_image_with_crop_or_pad(clean_img,220,220)
    #homo_size=[48,8]
    vst_clean = vst_f(a, b, clean_img_1)
    blur_clean = blur_0(clean_img,homo[:,0:m_num,:],m_num,bpg)
    #print("172!")
    # add noise to vst img
    # vst_clean = new_vst_f(a, b, clean_img)
    vst_clean_blur = vst_f(a, b, blur_clean)

    n = tf.random_normal(shape=tf.shape(vst_clean), stddev=1.0, name="noise_0")
    vst_noise_data = vst_clean_blur + n
    avg = vst_noise_data

    for i in range(frm_num-1): 
        blur_tmp=blur_0(clean_img,homo[:,i+1:i+1+m_num,:],m_num,bpg)
        vst_tmp = vst_f(a,b,blur_tmp)
        noise_tmp = tf.random_normal(shape=tf.shape(vst_clean), stddev=1.0, name="noise_%d"%(i+1))
        noise_data_tmp =  vst_tmp + noise_tmp
        avg = avg + noise_data_tmp
        vst_noise_data = tf.concat([vst_noise_data, noise_data_tmp], axis=-1)
    avg = avg / frm_num

    return vst_noise_data, vst_clean, avg


def blur_1(img,psf,blur_shape):
    psf_f = tf.stack([psf,psf,psf,psf],axis=-1)
    psf_final = tf.stack([psf_f],axis=-1)
    
    blur_img = tf.nn.depthwise_conv2d(img,psf_final,[1,1,1,1],padding='SAME')
    blur_img = tf.image.resize_image_with_crop_or_pad(blur_img,220,220)
    return blur_img

def preprocess_1(clean_img, bit_depth, iso, frm_num,psf):

    a, b = noise_profile_v2_voga(iso)
    clean_img = tf.cond(tf.greater(tf.random_normal([1])[0], 0), lambda: tf.transpose(clean_img,[0,2,1,3]), lambda: clean_img)
    clean_img = tf.cond(tf.greater(tf.random_normal([1])[0], 0), lambda: tf.reverse(clean_img,axis=[1]), lambda: clean_img) #verticle flipping
    clean_img = tf.cond(tf.greater(tf.random_normal([1])[0], 0), lambda: tf.reverse(clean_img,axis=[2]), lambda: clean_img)#herizontal flipping

    
    clean_img = tf.cast(clean_img, tf.float32)
    psf = tf.cast(psf,tf.float32)

    zeros = tf.zeros(tf.shape(clean_img), tf.float32)
    clean_img = tf.where(tf.greater(clean_img, zeros), clean_img, zeros)
        
    clean_img = clean_img / float(pow(2, bit_depth) - 1)
    k_size=[64,64]
    clean_img_1 = tf.image.resize_image_with_crop_or_pad(clean_img,220,220)
    vst_clean = vst_f(a, b, clean_img_1)
    blur_clean = blur_1(clean_img,psf[0],k_size)
    # add noise to vst img
    # vst_clean = new_vst_f(a, b, clean_img)
    vst_clean_blur = vst_f(a, b, blur_clean)

    n = tf.random_normal(shape=tf.shape(vst_clean), stddev=1.0, name="noise_0")
    vst_noise_data = vst_clean_blur + n
    avg = vst_noise_data

    for i in range(frm_num-1): 
        blur_tmp=blur_1(clean_img,psf[i],[64,64])
        
        vst_tmp = vst_f(a,b,blur_tmp)
        noise_tmp = tf.random_normal(shape=tf.shape(vst_clean), stddev=1.0, name="noise_%d"%(i+1))
        noise_data_tmp = noise_tmp + vst_tmp
        avg = avg + noise_data_tmp
        vst_noise_data = tf.concat([vst_noise_data, noise_data_tmp], axis=-1)
    avg = avg / frm_num

    return vst_noise_data, vst_clean, avg





def gl_loss(img1,img2):
        # total variation denoising
        shape = tuple(img1.get_shape().as_list())

        
        gl_loss = tf.reduce_mean(tf.abs((img1[:,1:,:,:] - img1[:,:shape[1]-1,:,:])  -(img2[:,1:,:,:] - img2[:,:shape[1]-1,:,:]))) + tf.reduce_mean(tf.abs((img1[:,:,1:,:] - img1[:,:,:shape[2]-1,:])-(img2[:,:,1:,:] - img2[:,:,:shape[2]-1,:])))

        return gl_loss





def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads



def train(record_list, h_list,psf_list,ckpt_dir, log_dir, batch_size=16, lr_init=3e-7, beta1=0.9, gpu_num=1, finetune=True, iter_start=275000):
    dataset = tf.data.TFRecordDataset(filenames=record_list)
    dataset = dataset.map(parse_example)
    dataset = dataset.shuffle(1000)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.repeat()
    
    iterator = dataset.make_one_shot_iterator()
    clean_img = iterator.get_next()
    #print(clean_img.shape,'clean_img')
    
    
    
    hset = tf.data.TFRecordDataset(filenames=h_list)
    hset = hset.map(parse_h)
    hset = hset.shuffle(10)
    hset = hset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    hset = hset.repeat()   
    
    hiterator = hset.make_one_shot_iterator()
    homo = hiterator.get_next()
    #print(homo, 'shape1')
    
    psfset = tf.data.TFRecordDataset(filenames=psf_list)
    psfset = psfset.map(parse_psf)
    psfset = psfset.shuffle(1000)
    psfset = psfset.apply(tf.contrib.data.batch_and_drop_remainder(4))
    psfset = psfset.repeat()
    psfiterator = psfset.make_one_shot_iterator()
    psf = psfiterator.get_next()
    
    
    n_iter = iter_start + 1
    vst_noise_data, vst_clean, vst_avg = tf.cond(tf.greater(n_iter%2, 0), lambda: preprocess_1(clean_img, 14, 25600, frm_num, psf), lambda: preprocess_0(clean_img, 14, 25600, frm_num, homo, m_num, batch_size)) #
    label_splits = tf.split(vst_clean, gpu_num,axis=0)
    #label_splits = tf.split(label_splits, 1,axis=-1)
    data_splits = tf.split(vst_noise_data, gpu_num,axis=0)
   # data_splits = tf.split(data_splits, frm_num,axis=-1)
 #   vst_avg_splits = tf.split(vst_avg, 1)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1)
    global_step = tf.Variable(0, name="global_step", trainable=False)

    reuse = False
    tower_grads = []
    loss_list = []
    #print("263!")
    
    with tf.variable_scope(tf.get_variable_scope()) as initScope:
        for i in range(gpu_num):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scpoe:
                    #print(data_splits[i].shape,'data_split')
                    out_image= PICNN_V1_4(tf.split(data_splits[i],frm_num,axis=-1),frm_num,reuse=reuse)
                    mse_loss = tl.cost.mean_squared_error(out_image, label_splits[i], is_mean=True)
                    #tf.get_variable_scope().reuse_variables()
                    #loss_fil=gl_loss(out_image,label_splits[i])
                    G_loss=mse_loss#tf.reduce_mean(tf.abs(out_image - label_splits[i]))*0.1+ loss_fil#+0.5*mse_loss#
                    tf.get_variable_scope().reuse_variables()
                    loss_list.append(G_loss)
                    grads = g_optim_init.compute_gradients(G_loss)
                    #loss_list.append(mse_loss)
                    #grads = g_optim_init.compute_gradients(mse_loss)
                    tower_grads.append(grads)
                    reuse = True
    mean_loss = tf.reduce_mean(loss_list)
    grads = average_gradients(tower_grads)
    apply_gradient_op = g_optim_init.apply_gradients(grads, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=1000 )
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    #tl.layers.initialize_global_variables(sess)
    #print("290!")


    total_loss = 0
    n_iter = iter_start
    if finetune == True:
        saver.restore(sess, ckpt_dir + 'ckpt-' + str(n_iter))

    # summary show in tensorboard
    # tf.summary.image("label", label_splits[-1][0:1, :, :, 1:2])
    # tf.summary.image("noide_data", data_splits[-1][0:1, :, :, 1:2])
    # tf.summary.image("net_out", net_out[0:1, :, :, 1:2])
    tf.summary.scalar('loss', mean_loss)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph, flush_secs=60)

    start_time = time.time()
    for i in range(200001):
        #errM, _ = sess.run([mean_loss, apply_gradient_op])
        #print("309!")
        #print("i%d!"%i)
        if n_iter % 2 == 0:
            use_psf = True
        else:
            use_psf = False
        if n_iter % 10000 == 1:   #test
            tmp_gt, tmp_in, tmp_res, errM, _ = sess.run(['split:0', 'split_1:0','GPU_0/PICNN_V1_4/layer41/BiasAdd:0', mean_loss,apply_gradient_op])
            dirr = FLAGS.tmp_res_dir + 'train_step_%07d' % n_iter
            create_dir(dirr)
            save_tmp_res(tmp_in[0, ...], tmp_gt[0, ...], tmp_res[0, ...], dirr)
            
        else:
            #print("317!")
            errM, _ = sess.run(
                [ mean_loss, apply_gradient_op])
        total_loss += errM

        if n_iter % 500 == 0:
            print('every 500 times iteration, the avg mse_loss is %f' %(total_loss/(i)))
            saver.save(sess, ckpt_dir+'ckpt', global_step=n_iter)
        if n_iter % 100 ==0:
            print("iter: %4d time: %4.4fs, mse: %.8f " % (n_iter, time.time() - start_time, errM))
            summary_result = sess.run(merged_summary_op)
            summary_writer.add_summary(summary_result, global_step=n_iter)
            start_time = time.time()
        n_iter += 1
        #print("331!")
#first change lv to 4e-6, then add mse loss

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    gpu_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    print("use %d gpu"%gpu_num)

    batch_size = 12 #16
    record_dir = "/DATA1/03.dlnr_tfrecord/02.bm3d_label_new_D2/"
    h_dir = "/DATA1/dlnr_zcz/homo/crop/train_4/"
    psf_dir = "/DATA1/dlnr_zcz/psf/"
    ckpt_dir = "/DATA1/dlnr_zcz/ckpt/20190118/1/"
    #ckpt_dir = "/DATA1/dlnr_zcz/ckpt/20190108/1/"
    #ckpt_dir = "/DATA1/dlnr_zcz/ckpt/20181228/4_1/"
    #ckpt_dir = "/DATA1/dlnr_zcz/ckpt/20190108/5/"
    #ckpt_dir = "/DATA1/dlnr_zcz/ckpt/20181226/4/"
    #log_dir = "/DATA1/dlnr_zcz/log/20181228/4_1/"
    log_dir = "/DATA1/dlnr_zcz/log/20190108/1/"
    record_list = glob.glob(record_dir+"*.tfrecord")
    h_list = glob.glob(h_dir+"*.tfrecord")
    psf_list = glob.glob(psf_dir+"*.tfrecord")


    create_dir(ckpt_dir)
    create_dir(log_dir)
    create_dir(FLAGS.tmp_res_dir)
    train(record_list, h_list, psf_list, ckpt_dir, log_dir, batch_size=batch_size, gpu_num=gpu_num)

