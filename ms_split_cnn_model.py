"""
Aaron Tun
MS Project: Split CNN Model 
SID: 008855652
"""
import os
# -----------------------------------------------------------------------------
# The following blocks of code separated by the 3 “# ++++” lines were taken 
# from multiple tutorials provided by Google on how to connect a TPU machine on 
# COLAB.  
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
IS_COLAB_BACKEND = 'COLAB_GPU' in os.environ  
                                             
if IS_COLAB_BACKEND:
  from google.colab import auth
  # Authenticates the Colab machine and TPU using 
  # credentials so that they can access the private GCS buckets.
  auth.authenticate_user()

# -----------------------------------------------------------------------------
import tensorflow as tf
print("Tensorflow version " + tf.__version__)
try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
  raise BaseException(
      'ERROR: Not connected to a TPU runtime;' 
      'please see the previous cell in this notebook for instructions!')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)

print("All devices: ", tf.config.list_logical_devices('TPU'))
print("Number of devices: ", len(tf.config.list_logical_devices('TPU')))
from tensorflow.python.profiler import profiler_client

tpu_profile_service_address = os.environ['COLAB_TPU_ADDR'].replace('8470', 
                                                                   '8466')
print(profiler_client.monitor(tpu_profile_service_address, 100, 2))
print('Number of devices: {}'.format(tpu_strategy.num_replicas_in_sync))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from glob import glob
import cv2 as opcv2
from tensorflow.keras.optimizers import SGD, Adam
from skimage.io import imread, imshow
from google.colab import files
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/')
import cnn_fun as cnn

# -----------------------------------------------------------------------------
# Functions to load and decode tfrecord files
def bytestring_to_pixels(record):

    img_size = 256
    name_to_features = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    parse_sample = tf.io.parse_example(record, name_to_features)

    byte_string = parse_sample['image_raw']
    image = tf.io.decode_image(byte_string, dtype=tf.bfloat16)
    image = tf.reshape(image, [800, 800, 3])

    image = tf.image.resize(image, size=(img_size, img_size))
    return image

def load_and_extract_img(filepath, data_type='Test or Train'):
  if data_type=='Train':
    """Return train dataset"""
    dataset = tf.data.Dataset.list_files(
                                  filepath,
                                  shuffle=True,
                                  seed=221,
                                ).interleave(
                                  tf.data.TFRecordDataset,
                                  cycle_length=tf.data.AUTOTUNE,
                                  num_parallel_calls=tf.data.AUTOTUNE
                                ).map(
                                  bytestring_to_pixels, 
                                  num_parallel_calls=tf.data.AUTOTUNE)
    return dataset
    
  elif data_type=='Test':
    """Return test dataset"""
    dataset = tf.data.Dataset.list_files(
                                  filepath,
                                  shuffle=False,
                                  seed=221,
                                ).interleave(
                                  tf.data.TFRecordDataset,
                                  cycle_length=tf.data.AUTOTUNE,
                                  num_parallel_calls=tf.data.AUTOTUNE
                                ).map(
                                  bytestring_to_pixels, 
                                  num_parallel_calls=tf.data.AUTOTUNE)
    return dataset
  
  else:
    print('Pick data type')

# -----------------------------------------------------------------------------
def ConvertBackTo255(img):
  """Convert image to RGB values of 0 to 255"""
  return opcv2.normalize(img, None, alpha=0, beta=255, 
                           norm_type=opcv2.NORM_MINMAX, dtype=opcv2.CV_8U)

# Load Train tfRecord Data
# -----------------------------------------------------------------------------
x1 = load_and_extract_img(
              'gs://ms_cfd_images/Train3_tfrecords/e_train/*.tfrecords', 
              data_type='Train') 
x2 = load_and_extract_img(
              'gs://ms_cfd_images/Train3_tfrecords/v_train/*.tfrecords', 
              data_type='Train') 
x3 = load_and_extract_img(
              'gs://ms_cfd_images/Train3_tfrecords/p_train/*.tfrecords', 
              data_type='Train') 
# -----------------------------------------------------------------------------
y1 = load_and_extract_img(
              'gs://ms_cfd_images/Train3_tfrecords/e_ground/*.tfrecords', 
              data_type='Train') 
y2 = load_and_extract_img(
              'gs://ms_cfd_images/Train3_tfrecords/v_ground/*.tfrecords', 
              data_type='Train') 
y3 = load_and_extract_img(
              'gs://ms_cfd_images/Train3_tfrecords/p_ground/*.tfrecords', 
              data_type='Train') 
# -----------------------------------------------------------------------------
xData = {'e_train': x1, 'v_train': x2, 'p_train': x3}
yData = {'e_ground': y1, 'v_ground': y2, 'p_ground': y3}
train_dataset = tf.data.Dataset.zip((xData, yData))

# Load Test tfRecord Data
# -----------------------------------------------------------------------------
x1_t = load_and_extract_img(
              'gs://ms_cfd_images/Test2_tfrecords/e_train/*.tfrecords', 
              data_type='Test') 
x2_t = load_and_extract_img(
              'gs://ms_cfd_images/Test2_tfrecords/v_train/*.tfrecords', 
              data_type='Test') 
x3_t = load_and_extract_img(
              'gs://ms_cfd_images/Test2_tfrecords/p_train/*.tfrecords', 
              data_type='Test') 
# -----------------------------------------------------------------------------
y1_t = load_and_extract_img(
              'gs://ms_cfd_images/Test2_tfrecords/e_ground/*.tfrecords', 
              data_type='Test') 
y2_t = load_and_extract_img(
              'gs://ms_cfd_images/Test2_tfrecords/v_ground/*.tfrecords', 
              data_type='Test') 
y3_t = load_and_extract_img(
              'gs://ms_cfd_images/Test2_tfrecords/p_ground/*.tfrecords', 
              data_type='Test') 
# -----------------------------------------------------------------------------
xData_t = {'e_train': x1_t, 'v_train': x2_t, 'p_train': x3_t}
yData_t = {'e_ground': y1_t, 'v_ground': y2_t, 'p_ground': y3_t}
test_data = tf.data.Dataset.zip((xData_t, yData_t))

# Declare variables
# -----------------------------------------------------------------------------
seed = 221
len_data = 7500
epochs = 20
batch_size = 64
buffer_size = len_data
repeat_size = 30
# -----------------------------------------------------------------------------
train_split = 0.90
val_split = 0.10
train_size = int(len_data*train_split)
val_size = int(len_data*val_split)

# -----------------------------------------------------------------------------
steps_per_epoch = (train_size*repeat_size) // batch_size
validation_steps = (val_size) // batch_size
# -----------------------------------------------------------------------------
print('Steps per epoch: {}'.format(steps_per_epoch))
print('Steps per validation: {}'.format(validation_steps))
print('Number of equivalent epochs: {}'.format(repeat_size*epochs))

# Set datasets
# -----------------------------------------------------------------------------
train_dat = train_dataset.take(
    train_size).cache().shuffle(buffer_size=buffer_size, seed=seed).repeat(
    ).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
# -----------------------------------------------------------------------------           
val_dat = train_dataset.skip(
    train_size).cache().shuffle(buffer_size=buffer_size, seed=seed).repeat(
    ).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
# -----------------------------------------------------------------------------
test_dat = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE) 
# -----------------------------------------------------------------------------
print('train set: {}'.format(train_size))
print('val set: {}'.format(val_size))
print('test set: {}'.format(834))

# Learning Rate for Run 1
diff_lr = 8e-4 - 8e-6
slope_lr = diff_lr/(steps_per_epoch*95)

run1_steps = 20*steps_per_epoch
run1_lr = 8e-4
run1_lr_end = run1_lr - slope_lr*run1_steps

print('Run 1 Learning Rate Starts at: {}, '\
      'with steps of {}, ends at {}'.format(run1_lr, 
                                            run1_steps,
                                            run1_lr_end))

# 2D CNN Model Split Inputs
# -----------------------------------------------------------------------------
img_size = 256
IMG_WIDTH = img_size
IMG_HIGHT = img_size
IMG_CH = 3
# Build ML Model
# -----------------------------------------------------------------------------
with tpu_strategy.scope():
# -----------------------------------------------------------------------------
  inputs_u = tf.keras.layers.Input(shape=(IMG_HIGHT, IMG_WIDTH, IMG_CH), 
                                                          name="e_train")
  inputs_v = tf.keras.layers.Input(shape=(IMG_HIGHT, IMG_WIDTH, IMG_CH), 
                                                          name="v_train")
  inputs_p = tf.keras.layers.Input(shape=(IMG_HIGHT, IMG_WIDTH, IMG_CH), 
                                                          name="p_train")
# -----------------------------------------------------------------------------
  kernel = 3
  fil_1 = 128
  fil_2 = 128
  fil_3 = 128
  fil_4 = 256
  fil_5 = 512
  fil_6 = 1024
  fil_7 = 1024
  fil_center = 2048
# -----------------------------------------------------------------------------
  # Contraction Blocks
# -----------------------------------------------------------------------------
  # Eddy Convolution Blocks
  u1, cu1 = cnn.ConvBlock(inputs_u, fil_1, kernel,
                          dropout=0.10, cnn_type='Contraction')
  u2, cu2 = cnn.ConvBlock(u1, fil_2, kernel, 
                          dropout=0.15, cnn_type='Contraction')
  u3, cu3 = cnn.ConvBlock(u2, fil_3, kernel, 
                          dropout=0.15, cnn_type='Contraction')
  u4, cu4 = cnn.ConvBlock(u3, fil_4, kernel, 
                          dropout=0.20, cnn_type='Contraction')
  u5, cu5 = cnn.ConvBlock(u4, fil_5, kernel, 
                          dropout=0.20, cnn_type='Contraction')
  u6, cu6 = cnn.ConvBlock(u5, fil_6, kernel, 
                          dropout=0.25, cnn_type='Contraction')
  u7, cu7 = cnn.ConvBlock(u6, fil_7, kernel, 
                          dropout=0.25, cnn_type='Contraction')
# -----------------------------------------------------------------------------
  # Velocity Convolution Blocks
  v1, cv1 = cnn.ConvBlock(inputs_v, fil_1, kernel, 
                          dropout=0.10, cnn_type='Contraction')
  v2, cv2 = cnn.ConvBlock(v1, fil_2, kernel, 
                          dropout=0.15, cnn_type='Contraction')
  v3, cv3 = cnn.ConvBlock(v2, fil_3, kernel, 
                          dropout=0.15, cnn_type='Contraction')
  v4, cv4 = cnn.ConvBlock(v3, fil_4, kernel, 
                          dropout=0.20, cnn_type='Contraction')
  v5, cv5 = cnn.ConvBlock(v4, fil_5, kernel, 
                          dropout=0.20, cnn_type='Contraction')
  v6, cv6 = cnn.ConvBlock(v5, fil_6, kernel, 
                          dropout=0.25, cnn_type='Contraction')
  v7, cv7 = cnn.ConvBlock(v6, fil_7, kernel, 
                          dropout=0.25, cnn_type='Contraction')
# -----------------------------------------------------------------------------
  # Pressure Convolution Blocks
  p1, cp1 = cnn.ConvBlock(inputs_p, fil_1, kernel, 
                          dropout=0.10, cnn_type='Contraction')
  p2, cp2 = cnn.ConvBlock(p1, fil_2, kernel, 
                          dropout=0.15, cnn_type='Contraction')
  p3, cp3 = cnn.ConvBlock(p2, fil_3, kernel, 
                          dropout=0.15, cnn_type='Contraction')
  p4, cp4 = cnn.ConvBlock(p3, fil_4, kernel, 
                          dropout=0.20, cnn_type='Contraction')
  p5, cp5 = cnn.ConvBlock(p4, fil_5, kernel, 
                          dropout=0.20, cnn_type='Contraction')
  p6, cp6 = cnn.ConvBlock(p5, fil_6, kernel, 
                          dropout=0.25, cnn_type='Contraction')
  p7, cp7 = cnn.ConvBlock(p6, fil_7, kernel, 
                          dropout=0.25, cnn_type='Contraction')
# -----------------------------------------------------------------------------
  # Center Convolution Block
# -----------------------------------------------------------------------------
  cat = cnn.ConvBlock(u7, fil_center, kernel, 
                          cnn_type='Concatenate', 
                          inputs2 = v7, 
                          inputs3 = p7)
  center = cnn.ConvBlock(cat, fil_center, kernel,
                          dropout=0.30, 
                          cnn_type='CenterBlock')
# -----------------------------------------------------------------------------
  # Expansion Block
# -----------------------------------------------------------------------------
  # Eddy Convolution Blocks
  u9 = cnn.ConvBlock(center, fil_7, kernel, 
                          dropout=0.25, cnn_type='Expansion', inputs2=cu7)
  u10 = cnn.ConvBlock(u9, fil_6, kernel, 
                          dropout=0.25, cnn_type='Expansion', inputs2=cu6)
  u11 = cnn.ConvBlock(u10, fil_5, kernel, 
                          dropout=0.20, cnn_type='Expansion', inputs2=cu5)
  u12 = cnn.ConvBlock(u11, fil_4, kernel, 
                          dropout=0.20, cnn_type='Expansion', inputs2=cu4)
  u13 = cnn.ConvBlock(u12, fil_3, kernel, 
                          dropout=0.15, cnn_type='Expansion', inputs2=cu3)
  u14 = cnn.ConvBlock(u13, fil_2, kernel, 
                          dropout=0.15, cnn_type='Expansion', inputs2=cu2)
  u15 = cnn.ConvBlock(u14, fil_1, kernel, 
                          dropout=0.10, cnn_type='Expansion', inputs2=cu1)
# -----------------------------------------------------------------------------
  # Velocity Convolution Blocks
  v9 = cnn.ConvBlock(center, fil_7, kernel, 
                          dropout=0.25, cnn_type='Expansion', inputs2=cv7)
  v10 = cnn.ConvBlock(v9, fil_6, kernel, 
                          dropout=0.25, cnn_type='Expansion', inputs2=cv6)
  v11 = cnn.ConvBlock(v10, fil_5, kernel, 
                          dropout=0.20, cnn_type='Expansion', inputs2=cv5)
  v12 = cnn.ConvBlock(v11, fil_4, kernel, 
                          dropout=0.20, cnn_type='Expansion', inputs2=cv4)
  v13 = cnn.ConvBlock(v12, fil_3, kernel, 
                          dropout=0.15, cnn_type='Expansion', inputs2=cv3)
  v14 = cnn.ConvBlock(v13, fil_2, kernel, 
                          dropout=0.15, cnn_type='Expansion', inputs2=cv2)
  v15 = cnn.ConvBlock(v14, fil_1, kernel, 
                          dropout=0.10, cnn_type='Expansion', inputs2=cv1)
# -----------------------------------------------------------------------------
  # Pressure Convolution Blocks
  p9 = cnn.ConvBlock(center, fil_7, kernel, 
                          dropout=0.25, cnn_type='Expansion', inputs2=cp7)
  p10 = cnn.ConvBlock(p9, fil_6, kernel, 
                          dropout=0.25, cnn_type='Expansion', inputs2=cp6)
  p11 = cnn.ConvBlock(p10, fil_5, kernel, 
                          dropout=0.20, cnn_type='Expansion', inputs2=cp5)
  p12 = cnn.ConvBlock(p11, fil_4, kernel, 
                          dropout=0.20, cnn_type='Expansion', inputs2=cp4)
  p13 = cnn.ConvBlock(p12, fil_3, kernel, 
                          dropout=0.15, cnn_type='Expansion', inputs2=cp3)
  p14 = cnn.ConvBlock(p13, fil_2, kernel, 
                          dropout=0.15, cnn_type='Expansion', inputs2=cp2)
  p15 = cnn.ConvBlock(p14, fil_1, kernel, 
                          dropout=0.10, cnn_type='Expansion', inputs2=cp1)
# -----------------------------------------------------------------------------
  # Outputs
# -----------------------------------------------------------------------------
  outputs_u = tf.keras.layers.Conv2D(3, (1, 1), activation='tanh', 
                                              name="e_ground")(u15)
  outputs_v = tf.keras.layers.Conv2D(3, (1, 1), activation='tanh', 
                                              name="v_ground")(v15)
  outputs_p = tf.keras.layers.Conv2D(3, (1, 1), activation='tanh', 
                                              name="p_ground")(p15)
# -----------------------------------------------------------------------------
  model = tf.keras.Model(inputs=[inputs_u, inputs_v, inputs_p], 
                         outputs=[outputs_u, outputs_v, outputs_p])

  lr_fn = tf.optimizers.schedules.PolynomialDecay(run1_lr, 
                                                  run1_steps, 
                                                  run1_lr_end, 1)
  adam = Adam(learning_rate=lr_fn)
  # adam = Adam(learning_rate=0.0008)
# -----------------------------------------------------------------------------
  model.compile(optimizer=adam,
                steps_per_execution = steps_per_epoch,
                loss=[tf.keras.losses.Huber(1.0),
                      tf.keras.losses.Huber(1.0),
                      tf.keras.losses.Huber(1.0)], 
                loss_weights=[1,1,1], 
                metrics=['accuracy'])
# -----------------------------------------------------------------------------
model.summary()
tf.keras.utils.plot_model(model, "CFD 3 Dense Bridge CNN.png", 
                                               show_shapes=True)

tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir='gs://ms_cfd_images/Tensorboard_2_1/', 
    histogram_freq=1, 
    update_freq='epoch')

# -----------------------------------------------------------------------------
# Remove Older Tensorboard Files 
!gsutil -m rm -r gs://ms_cfd_images/Tensorboard_2/

# Install Tensorboard Plugin
!pip install -U pip install -U tensorboard_plugin_profile

# Load the TensorBoard notebook extension.
%load_ext tensorboard

# Get TPU profiling service address. 
service_addr = tpu.get_master().replace(':8470', ':8466')
print(service_addr)

# Launch TensorBoard.
%tensorboard --logdir /content/Tensorboard_2_1

# -----------------------------------------------------------------------------
# Run 1
model.fit(train_dat,
          epochs=epochs,
          initial_epoch=0,
          steps_per_epoch=steps_per_epoch,
          validation_data=val_dat,
          validation_steps=validation_steps,
          callbacks=[tb_callback])

# Run 1
model.save('Model_2D_499mil_1.h5')

# Send Run 1 to GCS
bucket_name_cnn = "ms_cfd_images/cnn_models"
!gsutil -m cp /content/Model_2D_499mil_1.h5 gs://{bucket_name_cnn}

# -----------------------------------------------------------------------------
# Adjust Learning Rate for Run 2
epochs = 60
run2_steps = 40*steps_per_epoch
run2_lr_start = run1_lr_end
run2_lr_end = run2_lr_start - slope_lr*run2_steps

print('Run 2 Learning Rate Starts at: {}, '\
      'with steps of {}, ends at {}'.format(run2_lr_start, 
                                            run2_steps,
                                            run2_lr_end))

# Run 2
with tpu_strategy.scope(): 
  model = tf.keras.models.load_model(
      'gs://ms_cfd_images/cnn_models/Model_2D_499mil_1.h5')
  lr_fn = tf.optimizers.schedules.PolynomialDecay(run2_lr_start, 
                                                  run2_steps, 
                                                  run2_lr_end, 1)
  adam = Adam(learning_rate=lr_fn)
  # ---------------------------------------------------------------------------
  model.compile(optimizer=adam,
                steps_per_execution = steps_per_epoch,
                loss=[tf.keras.losses.Huber(1.0),
                      tf.keras.losses.Huber(1.0),
                      tf.keras.losses.Huber(1.0)],
                loss_weights=[1,1,1], 
                metrics=['accuracy'])

# Run 2
model.fit(train_dat,
          epochs=epochs,
          initial_epoch=20,
          steps_per_epoch=steps_per_epoch,
          validation_data=val_dat,
          validation_steps=validation_steps,
          callbacks=[tb_callback])

# Run 2
model.save('Model_2D_499mil_2.h5')

# Send Run 2 to GCS
bucket_name_cnn = "ms_cfd_images/cnn_models"
!gsutil -m cp /content/Model_2D_499mil_2.h5 gs://{bucket_name_cnn}

# -----------------------------------------------------------------------------
# Adjust Learning Rate for Run 3
epochs = 100
run3_steps = 35*steps_per_epoch
run3_lr_start = run2_lr_end
run3_lr_end = run2_lr_end - slope_lr*run3_steps

print('Run 3 Learning Rate Starts at: {}, '\
      'with steps of {}, ends at {}'.format(run3_lr_start, 
                                            run3_steps,
                                            run3_lr_end))

# Run 3
with tpu_strategy.scope(): 
  model = tf.keras.models.load_model(
      'gs://ms_cfd_images/cnn_models/Model_2D_499mil_2.h5')
  lr_fn = tf.optimizers.schedules.PolynomialDecay(run3_lr_start, 
                                                  run3_steps, 
                                                  run3_lr_end, 1)
  adam = Adam(learning_rate=lr_fn)
  # ---------------------------------------------------------------------------
  model.compile(optimizer=adam,
                steps_per_execution = steps_per_epoch,
                loss=[tf.keras.losses.Huber(1.0),
                      tf.keras.losses.Huber(1.0),
                      tf.keras.losses.Huber(1.0)],
                loss_weights=[1,1,1], 
                metrics=['accuracy'])

# Run 3
model.fit(train_dat,
          epochs=epochs,
          initial_epoch=60,
          steps_per_epoch=steps_per_epoch,
          validation_data=val_dat,
          validation_steps=validation_steps,
          callbacks=[tb_callback])

# Run 3
model.save('Model_2D_499mil_3.h5')
# Send Model 3 to GCS

bucket_name_cnn = "ms_cfd_images/cnn_models"
!gsutil -m cp /content/Model_2D_499mil_3.h5 gs://{bucket_name_cnn}

# -----------------------------------------------------------------------------
# Load Final Model and Evaluate/Predict Model
with tpu_strategy.scope(): 
  model = tf.keras.models.load_model(
      'gs://ms_cfd_images/cnn_models/Model_2D_499mil_3.h5')

model.evaluate(test_dat, steps=14)

preds_train_u, preds_train_v, preds_train_p = model.predict(test_dat, steps=14)

preds_train_u = ConvertBackTo255(preds_train_u)
preds_train_v = ConvertBackTo255(preds_train_v)
preds_train_p = ConvertBackTo255(preds_train_p)

# -----------------------------------------------------------------------------
# Plot Results
index = 569
take_num = 10
range_from = 2
range_to = 3
airfoil = "2424"

# -----------------------------------------------------------------------------
# high eddy
i=index
for val in test_dat.take(take_num):
    for j in range(range_from, range_to):
        

        fig, axs = plt.subplots(3, 1, figsize = (10, 30))
        axs[0].imshow(val[0]['e_train'][j])
        axs[0].axis('off')
        print(i)
        axs[1].imshow(preds_train_u[i])
        axs[1].axis('off')
        axs[2].imshow(val[1]['e_ground'][j])
        axs[2].axis('off')

        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_e_hAoA_split.jpeg".format(airfoil), dpi = 100)
        plt.close()
        
        fig2 = plt.figure(figsize=(25,25))
        plt.imshow(preds_train_u[i])
        plt.axis('off')
        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_e_hAoA_split_preds.jpeg".format(airfoil), dpi = 100)
        plt.close()
         
        fig3 = plt.figure(figsize=(25,25))
        plt.imshow(val[1]['e_ground'][j])
        plt.axis('off')
        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_e_hAoA_split_gtruth.jpeg".format(airfoil), dpi = 100)
        plt.close()

        i+=1

# -----------------------------------------------------------------------------
# high pressure
i=index
for val in test_dat.take(take_num):
    for j in range(range_from, range_to):
        fig, axs = plt.subplots(3, 1, figsize = (10, 30))
        axs[0].imshow(val[0]['p_train'][j])
        axs[0].axis('off')
        print(i)
        axs[1].imshow(preds_train_p[i])
        axs[1].axis('off')
        axs[2].imshow(val[1]['p_ground'][j])
        axs[2].axis('off')

        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_p_hAoA_split.jpeg".format(airfoil), dpi = 100)
        plt.close()
        
        fig2 = plt.figure(figsize=(25,25))
        plt.imshow(preds_train_p[i])
        plt.axis('off')
        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_p_hAoA_split_preds.jpeg".format(airfoil), dpi = 100)
        plt.close()
         
        fig3 = plt.figure(figsize=(25,25))
        plt.imshow(val[1]['p_ground'][j])
        plt.axis('off')
        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_p_hAoA_split_gtruth.jpeg".format(airfoil), dpi = 100)
        plt.close()
        i+=1

# -----------------------------------------------------------------------------
# high velocity
i=index
for val in test_dat.take(take_num):
    for j in range(range_from, range_to):
        fig, axs = plt.subplots(3, 1, figsize = (10, 30))
        axs[0].imshow(val[0]['v_train'][j])
        axs[0].axis('off')
        print(i)
        axs[1].imshow(preds_train_v[i])
        axs[1].axis('off')
        axs[2].imshow(val[1]['v_ground'][j])
        axs[2].axis('off')

        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_v_hAoA_split.jpeg".format(airfoil), dpi = 100)
        plt.close()
        
        fig2 = plt.figure(figsize=(25,25))
        plt.imshow(preds_train_v[i])
        plt.axis('off')
        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_v_hAoA_split_preds.jpeg".format(airfoil), dpi = 100)
        plt.close()
         
        fig3 = plt.figure(figsize=(25,25))
        plt.imshow(val[1]['v_ground'][j])
        plt.axis('off')
        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_v_hAoA_split_gtruth.jpeg".format(airfoil), dpi = 100)
        plt.close()
        i+=1

# -----------------------------------------------------------------------------
index = 812
take_num = 13
range_from = 56
range_to = 57
airfoil = "2424"

# -----------------------------------------------------------------------------
# low eddy
i=index
for val in test_dat.take(take_num):
    for j in range(range_from, range_to):
        

        fig, axs = plt.subplots(3, 1, figsize = (10, 30))
        axs[0].imshow(val[0]['e_train'][j])
        axs[0].axis('off')
        print(i)
        axs[1].imshow(preds_train_u[i])
        axs[1].axis('off')
        axs[2].imshow(val[1]['e_ground'][j])
        axs[2].axis('off')

        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_e_lAoA_split.jpeg".format(airfoil), dpi = 100)
        plt.close()
        
        fig2 = plt.figure(figsize=(25,25))
        plt.imshow(preds_train_u[i])
        plt.axis('off')
        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_e_lAoA_split_preds.jpeg".format(airfoil), dpi = 100)
        plt.close()
         
        fig3 = plt.figure(figsize=(25,25))
        plt.imshow(val[1]['e_ground'][j])
        plt.axis('off')
        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_e_lAoA_split_gtruth.jpeg".format(airfoil), dpi = 100)
        plt.close()

        i+=1

# -----------------------------------------------------------------------------
# low pressure
i=index
for val in test_dat.take(take_num):
    for j in range(range_from, range_to):
        fig, axs = plt.subplots(3, 1, figsize = (10, 30))
        axs[0].imshow(val[0]['p_train'][j])
        axs[0].axis('off')
        print(i)
        axs[1].imshow(preds_train_p[i])
        axs[1].axis('off')
        axs[2].imshow(val[1]['p_ground'][j])
        axs[2].axis('off')

        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_p_lAoA_split.jpeg".format(airfoil), dpi = 100)
        plt.close()
        
        fig2 = plt.figure(figsize=(25,25))
        plt.imshow(preds_train_p[i])
        plt.axis('off')
        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_p_lAoA_split_preds.jpeg".format(airfoil), dpi = 100)
        plt.close()
         
        fig3 = plt.figure(figsize=(25,25))
        plt.imshow(val[1]['p_ground'][j])
        plt.axis('off')
        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_p_lAoA_split_gtruth.jpeg".format(airfoil), dpi = 100)
        plt.close()
        i+=1

# -----------------------------------------------------------------------------
# low velocity
i=index
for val in test_dat.take(take_num):
    for j in range(range_from, range_to):
        fig, axs = plt.subplots(3, 1, figsize = (10, 30))
        axs[0].imshow(val[0]['v_train'][j])
        axs[0].axis('off')
        print(i)
        axs[1].imshow(preds_train_v[i])
        axs[1].axis('off')
        axs[2].imshow(val[1]['v_ground'][j])
        axs[2].axis('off')

        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_v_lAoA_split.jpeg".format(airfoil), dpi = 100)
        plt.close()
        
        fig2 = plt.figure(figsize=(25,25))
        plt.imshow(preds_train_v[i])
        plt.axis('off')
        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_v_lAoA_split_preds.jpeg".format(airfoil), dpi = 100)
        plt.close()
         
        fig3 = plt.figure(figsize=(25,25))
        plt.imshow(val[1]['v_ground'][j])
        plt.axis('off')
        fig.tight_layout()
        plt.savefig("/content/drive/MyDrive/n{}".format(airfoil)+
                    "/n{}_v_lAoA_split_gtruth.jpeg".format(airfoil), dpi = 100)
        plt.close()
        i+=1