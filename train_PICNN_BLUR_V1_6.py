from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import glob
from skimage import filters
import math,cv2
import tensorlayer as tl
from model_zcz import *
from pre_funs import prep_np,save_test

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('tmp_res_dir', './picnn_blur_denoise/tmp/0702/', 'the tmp result during training ')

#ps = 256  # patch size for training  512
frm_num=4#6 #

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



def parse_psf(psf_proto):
    dics = {}
    # dics['label'] = tf.FixedLenFeature(shape=[],dtype=tf.int64)
    dics['psf'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
    dics['shape'] = tf.FixedLenFeature(shape=[2], dtype=tf.int64)
    dics['psf_num'] = tf.FixedLenFeature(shape=[1], dtype=tf.int64)
    parsed_psf = tf.parse_single_example(serialized=psf_proto,features=dics)

    psf = parsed_psf['psf']
    psf = tf.decode_raw(psf, tf.float64)

    psf = tf.reshape(psf, [32,32])#64,64
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

def blur(img,psf):
    psf_f = tf.Variable(tf.stack([tf.stack([psf,psf,psf,psf],axis=-1)],axis=-1),trainable=False,name="psf")
    #psf_final = tf.stack([psf_f],axis=-1)
    blur_img = tf.nn.depthwise_conv2d(img,psf_f,[1,1,1,1],padding='SAME')
    blur_img = tf.image.resize_image_with_crop_or_pad(blur_img,118,118)#134
    #mosiac
    blur_b = blur_img[:,0:118:2,0:118:2,0]#59*59#134*134
    blur_g1 = blur_img[:,0:118:2,1:118:2,1]
    blur_g2 = blur_img[:,1:118:2,0:118:2,2]#134*134
    blur_r = blur_img[:,1:118:2,1:118:2,3]

    blur_img = tf.stack((blur_b,blur_g1,blur_g2,blur_r),axis=-1)#59,59,4    
    
    return blur_img

def save_img(raw, dir, name):
    t_raw = np.clip(raw, 0, 1)
    t_raw = t_raw * 255 * 2
    t_raw = np.clip(t_raw, 0, 255).astype(np.uint8)
    for j in range(4):
        cv2.imwrite(dir + '/%s_%d.png' % (name, j), t_raw[:, :, j])


# save the tmp image during training
def save_tmp_res(tmp_in, tmp_gt, tmp_res, dirr):
    a, b = noise_profile_v2(25600)
    tmp_in = inv_vst_f_n(a,b,tmp_in)
    tmp_gt = inv_vst_f_n(a,b,tmp_gt)
    tmp_res = inv_vst_f_n(a,b,tmp_res)
    frame_num = frm_num
    for jj in range(frame_num):
        raw_tmp = tmp_in[:, :, 4*jj:4*jj+4]
        save_img(raw_tmp, dirr, 'noisy_%d' % jj)
    save_img(tmp_gt, dirr, 'gt')
    save_img(tmp_res, dirr, 'net_out')


def preprocess(clean_img, bit_depth, iso, frm_num,psf):
    # a, b = new_noise_profile(iso)
    # max_v = new_cal_max_v(a, b)

    # iso_index = tf.random_uniform(shape=[1], minval=0, maxval=2, dtype=tf.int32)
    # a, b = noise_profile_v2(tf.convert_to_tensor(iso_list)[iso_index])
    a, b = noise_profile_v2_voga(iso)
    #ds_size = 300#1100
    #clean_img = tf.random_crop(clean_img, [tf.shape(clean_img)[0], ds_size, ds_size, 4])
    clean_img = tf.cond(tf.greater(tf.random_normal([1])[0], 0), lambda: tf.transpose(clean_img,[0,2,1,3]), lambda: clean_img)
    clean_img = tf.cond(tf.greater(tf.random_normal([1])[0], 0), lambda: tf.reverse(clean_img,axis=[1]), lambda: clean_img) #verticle flipping
    clean_img = tf.cond(tf.greater(tf.random_normal([1])[0], 0), lambda: tf.reverse(clean_img,axis=[2]), lambda: clean_img)#herizontal flipping
    #clean_img = tf.image.random_flip_left_right(clean_img)
    #clean_img = tf.image.random_flip_up_down(clean_img)
    
    clean_img = tf.cast(clean_img, tf.float32)
    psf = tf.cast(psf,tf.float32)
    # blc = pow(2, bit_depth - 4)
    # clean_img = clean_img-blc
    zeros = tf.zeros(tf.shape(clean_img), tf.float32)
    clean_img = tf.where(tf.greater(clean_img, zeros), clean_img, zeros)
        
    clean_img = clean_img / float(pow(2, bit_depth) - 1)
    '''
    clean_img_1_0 = (clean_img[:,0:ds_size:2,0:ds_size:2,0]+clean_img[:,0:ds_size:2,1:ds_size:2,0]+clean_img[:,1:ds_size:2,0:ds_size:2,0]+clean_img[:,1:ds_size:2,1:ds_size:2,0])/4.0  #150*150
    clean_img_1_1 = (clean_img[:,0:ds_size:2,0:ds_size:2,1]+clean_img[:,0:ds_size:2,1:ds_size:2,1]+clean_img[:,1:ds_size:2,0:ds_size:2,1]+clean_img[:,1:ds_size:2,1:ds_size:2,1])/4.0   #150*150
    clean_img_1_2 = (clean_img[:,0:ds_size:2,0:ds_size:2,2]+clean_img[:,0:ds_size:2,1:ds_size:2,2]+clean_img[:,1:ds_size:2,0:ds_size:2,2]+clean_img[:,1:ds_size:2,1:ds_size:2,2])/4.0#
    clean_img_1_3 = (clean_img[:,0:ds_size:2,0:ds_size:2,3]+clean_img[:,0:ds_size:2,1:ds_size:2,3]+clean_img[:,1:ds_size:2,0:ds_size:2,3]+clean_img[:,1:ds_size:2,1:ds_size:2,3])/4.0
    
    #clean_img_g = (clean_img_1_1+clean_img_1_2)/2.0#150*150
    clean_img_1 = tf.stack((clean_img_1_0,clean_img_1_1,clean_img_1_2,clean_img_1_3),axis=-1)#150,150,3 #realistic photon   
    '''
    clean_img_1 = slim.avg_pool2d(clean_img, [2, 2], padding='SAME')#150
    clean_img_crop = tf.image.resize_image_with_crop_or_pad(clean_img_1,118,118)
    
    #gt mosiac
    clean_b = clean_img_crop[:,0:118:2,0:118:2,0]#59*59#67
    clean_g1 = clean_img_crop[:,0:118:2,1:118:2,1]
    clean_g2 = clean_img_crop[:,1:118:2,0:118:2,2]
    clean_r = clean_img_crop[:,1:118:2,1:118:2,3] 
    
    
    clean_img_2 =  tf.stack((clean_b,clean_g1,clean_g2,clean_r),axis=-1)   
    vst_clean = vst_f(a, b, clean_img_2)
    blur_clean = blur(clean_img_1,psf[0])
    # add noise to vst img
    vst_clean_blur = vst_f(a, b, blur_clean)

    n = tf.random_normal(shape=tf.shape(vst_clean), stddev=1.0, name="noise_0")
    vst_noise_data = vst_clean_blur + n
    avg = vst_noise_data

    for i in range(frm_num-1): 
        blur_tmp=blur(clean_img_1,psf[i+1])#59*59*4
        
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
        #gl_y_size = _tensor_size(image[:,1:,:,:])
        #gl_x_size = _tensor_size(image[:,:,1:,:])
        
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



def train(record_list, psf_list,ckpt_dir, log_dir, batch_size=16, lr_init=2e-5, beta1=0.9, gpu_num=1, finetune=True, iter_start=991500):
    dataset = tf.data.TFRecordDataset(filenames=record_list)
    dataset = dataset.map(parse_example)
    dataset = dataset.shuffle(1000)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.repeat()
    
    iterator = dataset.make_one_shot_iterator()
    clean_img = iterator.get_next()
    
    
    psfset = tf.data.TFRecordDataset(filenames=psf_list)
    psfset = psfset.map(parse_psf)
    psfset = psfset.shuffle(1000)
    psfset = psfset.apply(tf.contrib.data.batch_and_drop_remainder(frm_num))
    psfset = psfset.repeat()
    psfiterator = psfset.make_one_shot_iterator()
    psf = psfiterator.get_next()
    
    
    vst_noise_data, vst_clean, vst_avg = preprocess(clean_img, 14, 25600, frm_num, psf)
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
    
    
    with tf.variable_scope(tf.get_variable_scope()) as initScope:
        for i in range(gpu_num):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scpoe:
                    #print(data_splits[i].shape,'data_split')
                    out_image= PICNN_V1_4(tf.split(data_splits[i],frm_num,axis=-1),frm_num,reuse=reuse)
                    mse_loss = tl.cost.mean_squared_error(out_image, label_splits[i], is_mean=True)
                    #tf.get_variable_scope().reuse_variables()
                    loss_fil=gl_loss(out_image,label_splits[i])
                    G_loss=mse_loss#tf.reduce_mean(tf.abs(out_image - label_splits[i]))*0.1+ loss_fil#+mse_loss#
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
    
    ''' 
    ############################# quickly test#######################################
    raw_files = glob.glob('/DATA1/02.dlnr_test_data/src/08.cbg_sid_dpcc_v2/20180904_170218/' + '*.raw')
    raw_files.sort()
    img_test, avg = prep_np(raw_files, iso=24575, size=(2736, 3648))
    t_image = tf.placeholder('float32', [None, None, None, 24], name='input_image')
    tf.get_variable_scope().reuse_variables()
    net_g = PICNN_V1_4(t_image, 4, reuse=True)
    #################################################################################
    '''

    saver = tf.train.Saver(max_to_keep=1000 )
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    #tl.layers.initialize_global_variables(sess)

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
    for i in range(10001):
        #errM, _ = sess.run([mean_loss, apply_gradient_op])
        
        if n_iter % 10000 == 1:   #test
            tmp_gt, tmp_in, tmp_res, errM, _ = sess.run(['split:0', 'split_1:0','GPU_0/PICNN_V1_4/layer41/BiasAdd:0', mean_loss,apply_gradient_op])
            dirr = FLAGS.tmp_res_dir + 'train_step_%07d' % n_iter
            create_dir(dirr)
            save_tmp_res(tmp_in[0, ...], tmp_gt[0, ...], tmp_res[0, ...], dirr)
        else:
            errM, _ = sess.run(
                [ mean_loss, apply_gradient_op])
        total_loss += errM
        '''        
        if i % 10000 == 0:
            out_image = sess.run(net_g, {t_image: [img_test]})
            save_test(out_image, ckpt_dir+"img/%08d_deblur_denoised.png" % n_iter, iso=24575)
            #save_img(output, gt, ckpt_dir+"img/%08d_denoised_gan.png" % n_iter)
        '''            
        if n_iter % 500 == 0:
            print('every 500 times iteration, the avg mse_loss is %f' %(total_loss/i))
            saver.save(sess, ckpt_dir+'ckpt', global_step=n_iter)
        if n_iter % 100 ==0:
            print("iter: %4d time: %4.4fs, mse: %.8f " % (n_iter, time.time() - start_time, errM))
            summary_result = sess.run(merged_summary_op)
            summary_writer.add_summary(summary_result, global_step=n_iter)
            start_time = time.time()
        n_iter += 1
#first change lv to 4e-6, then add mse loss

def train_1(record_list, psf_list,ckpt_dir, log_dir, batch_size=16, lr_init=4e-5, beta1=0.9, gpu_num=1, finetune=True, iter_start=263500):
    dataset = tf.data.TFRecordDataset(filenames=record_list)
    dataset = dataset.map(parse_example)
    dataset = dataset.shuffle(1000)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.repeat()
    
    iterator = dataset.make_one_shot_iterator()
    clean_img = iterator.get_next()
    
    
    psfset = tf.data.TFRecordDataset(filenames=psf_list)
    psfset = psfset.map(parse_psf)
    psfset = psfset.shuffle(1000)
    psfset = psfset.apply(tf.contrib.data.batch_and_drop_remainder(frm_num))
    psfset = psfset.repeat()
    psfiterator = psfset.make_one_shot_iterator()
    psf = psfiterator.get_next()
    
    
    vst_noise_data, vst_clean, vst_avg = preprocess(clean_img, 14, 25600, frm_num, psf)
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
    
    
    with tf.variable_scope(tf.get_variable_scope()) as initScope:
        for i in range(gpu_num):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scpoe:
                    #print(data_splits[i].shape,'data_split')
                    out_image= PICNN_V1_5(data_splits[i],frm_num,reuse=reuse)
                    mse_loss = tl.cost.mean_squared_error(out_image, label_splits[i], is_mean=True)
                    #tf.get_variable_scope().reuse_variables()
                    #loss_fil=gl_loss(out_image,label_splits[i])
                    G_loss=mse_loss#tf.reduce_mean(tf.abs(out_image - label_splits[i]))*0.1+ loss_fil#+mse_loss#
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
    
    ''' 
    ############################# quickly test#######################################
    raw_files = glob.glob('/DATA1/02.dlnr_test_data/src/08.cbg_sid_dpcc_v2/20180904_170218/' + '*.raw')
    raw_files.sort()
    img_test, avg = prep_np(raw_files, iso=24575, size=(2736, 3648))
    t_image = tf.placeholder('float32', [None, None, None, 24], name='input_image')
    tf.get_variable_scope().reuse_variables()
    net_g = PICNN_V1_4(t_image, 4, reuse=True)
    #################################################################################
    '''

    saver = tf.train.Saver(max_to_keep=1000 )
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    #tl.layers.initialize_global_variables(sess)

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
    for i in range(500001):
        #errM, _ = sess.run([mean_loss, apply_gradient_op])
        
        if n_iter % 10000 == 1:   #test
            tmp_gt, tmp_in, tmp_res, errM, _ = sess.run(['split:0', 'split_1:0','GPU_0/generator/out_2/BiasAdd:0', mean_loss,apply_gradient_op])
            dirr = FLAGS.tmp_res_dir + 'train_step_%07d' % n_iter
            create_dir(dirr)
            save_tmp_res(tmp_in[0, ...], tmp_gt[0, ...], tmp_res[0, ...], dirr)
        else:
            errM, _ = sess.run(
                [ mean_loss, apply_gradient_op])
        total_loss += errM
        '''        
        if i % 10000 == 0:
            out_image = sess.run(net_g, {t_image: [img_test]})
            save_test(out_image, ckpt_dir+"img/%08d_deblur_denoised.png" % n_iter, iso=24575)
            #save_img(output, gt, ckpt_dir+"img/%08d_denoised_gan.png" % n_iter)
        '''            
        if n_iter % 500 == 0:
            print('every 500 times iteration, the avg mse_loss is %f' %(total_loss/i))
            saver.save(sess, ckpt_dir+'ckpt', global_step=n_iter)
        if n_iter % 100 ==0:
            print("iter: %4d time: %4.4fs, mse: %.8f " % (n_iter, time.time() - start_time, errM))
            summary_result = sess.run(merged_summary_op)
            summary_writer.add_summary(summary_result, global_step=n_iter)
            start_time = time.time()
        n_iter += 1
#first change lv to 4e-6, then add mse loss




if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    gpu_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    print("use %d gpu"%gpu_num)
    '''
    batch_size = 4
    record_dir = "/DATA1/03.dlnr_tfrecord/02.bm3d_label_new_D2/"
    psf_dir = "/DATA1/dlnr_zcz/psf_32(1)/"#64*64#16*16#32*32
    ckpt_dir = "/DATA1/dlnr_zcz/ckpt/20190618/1/"#17/1/"
    log_dir = "/DATA1/dlnr_zcz/log/20190618/1/"#6 frm,picnn_v1_5
    '''
    '''
    batch_size = 8
    record_dir = "/DATA1/03.dlnr_tfrecord/04.bm3d_label_new_D4/"
    psf_dir = "/DATA1/dlnr_zcz/psf/"#64*64#16*16#32*32
    ckpt_dir = "/DATA1/dlnr_zcz/ckpt/20190524/1/"#17/1/"
    log_dir = "/DATA1/dlnr_zcz/log/20190524/1/"#17/1/"
    '''
    '''
    batch_size = 8
    record_dir = "/DATA1/03.dlnr_tfrecord/02.bm3d_label_new_D2/"
    psf_dir = "/DATA1/dlnr_zcz/psf_32(1)/"#16*16#32*32
    ckpt_dir = "/DATA1/dlnr_zcz/ckpt/2019626/1/"#6frm,1000000,picnn_v1_4#17/1/"
    log_dir = "/DATA1/dlnr_zcz/log/20190626/1/"#17/1/"
    '''
    batch_size = 16
    record_dir = "/DATA1/03.dlnr_tfrecord/02.bm3d_label_new_D2/"
    psf_dir = "/DATA1/dlnr_zcz/psf_32(1)/"
    ckpt_dir = "/DATA1/dlnr_zcz/ckpt/20190702/"
    log_dir = "/DATA1/dlnr_zcz/log/20190702/"    
    
    record_list = glob.glob(record_dir+"*.tfrecord")
    psf_list = glob.glob(psf_dir+"*.tfrecord")
    

    create_dir(ckpt_dir)
    create_dir(log_dir)
    train_1(record_list, psf_list, ckpt_dir, log_dir, batch_size=batch_size, gpu_num=gpu_num)

