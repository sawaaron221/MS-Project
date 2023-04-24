"""
Aaron Tun
MS Project 
"""
import os
# -----------------------------------------------------------------------------
IS_COLAB_BACKEND = 'COLAB_GPU' in os.environ  
if IS_COLAB_BACKEND:
  from google.colab import auth
  auth.authenticate_user()
# -----------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import io
import PIL
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline

# ----------------------------------------------------------------------------- 
# Functions that convert images into binary format (tfrecord).
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def img_example(image):
    
    bytes_buffer = io.BytesIO()
    image.convert('RGB').save(bytes_buffer, "JPEG")
    image_bytes = bytes_buffer.getvalue()

    feature = {
        'image_raw': _bytes_feature(image_bytes),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# ----------------------------------------------------------------------------- 
# Functions to decode tfrecord files
def parse_record(record):
    name_to_features = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(record, name_to_features)

def bytestring_to_pixels(parse_sample):
    byte_string = parse_sample['image_raw']
    image = tf.io.decode_image(byte_string)
    image = tf.reshape(image, [800, 800, 3])
    return image

def load_and_extract_img(filepath):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(bytestring_to_pixels, 
                          num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# -----------------------------------------------------------------------------
# Create tfRecord Data
img_size = 256
IMG_WIDTH = img_size
IMG_HIGHT = img_size
IMG_CH = 3

TRAIN_PATH_eL = '/content/drive/MyDrive/'\
                'MS CFD Data/Train_Datasets/eddy/train/'
TRAIN_PATH_eH = '/content/drive/MyDrive/'\
                'MS CFD Data/Train_Datasets/eddy/ground_truth/'

TRAIN_PATH_vL = '/content/drive/MyDrive/'\
                'MS CFD Data/Train_Datasets/velocity/train/'
TRAIN_PATH_vH = '/content/drive/MyDrive/'\
                'MS CFD Data/Train_Datasets/velocity/ground_truth/'

TRAIN_PATH_pL = '/content/drive/MyDrive/'\
                'MS CFD Data/Train_Datasets/pressure/train/'
TRAIN_PATH_pH = '/content/drive/MyDrive/'\
                'MS CFD Data/Train_Datasets/pressure/ground_truth/'
# -----------------------------------------------------------------------------

train_e_idL = next(os.walk(TRAIN_PATH_eL))[2]
train_e_idH = next(os.walk(TRAIN_PATH_eH))[2]

train_v_idL = next(os.walk(TRAIN_PATH_vL))[2]
train_v_idH = next(os.walk(TRAIN_PATH_vH))[2]

train_p_idL = next(os.walk(TRAIN_PATH_pL))[2]
train_p_idH = next(os.walk(TRAIN_PATH_pH))[2]

train_e_idL.sort()
train_v_idL.sort()
train_p_idL.sort()
train_e_idH.sort()
train_v_idH.sort()
train_p_idH.sort()

# -----------------------------------------------------------------------------
# Split Data {Train}
# Train
indexs = [2, 19, 22, 39, 42, 59, 62, 79, 82, 99]
e_train = []
v_train = []
p_train = []
e_ground = []
v_ground = []
p_ground = []

# -----------------------------------------------------------------------------
e_train.extend(train_e_idL[0:83])
v_train.extend(train_v_idL[0:83])
p_train.extend(train_p_idL[0:83])

e_ground.extend(train_e_idH[0:83])
v_ground.extend(train_v_idH[0:83])
p_ground.extend(train_p_idH[0:83])

# -----------------------------------------------------------------------------
for i in range(0,len(indexs)-1):

  temp0 = train_e_idL[indexs[i]*83:(indexs[i+1]-1)*83]
  e_train.extend(temp0)
  temp1 = train_v_idL[indexs[i]*83:(indexs[i+1]-1)*83]
  v_train.extend(temp1)
  temp2 = train_p_idL[indexs[i]*83:(indexs[i+1]-1)*83]
  p_train.extend(temp2)

  temp3 = train_e_idH[indexs[i]*83:(indexs[i+1]-1)*83]
  e_ground.extend(temp3)
  temp4 = train_v_idH[indexs[i]*83:(indexs[i+1]-1)*83]
  v_ground.extend(temp4)
  temp5 = train_p_idH[indexs[i]*83:(indexs[i+1]-1)*83]
  p_ground.extend(temp5)

# -----------------------------------------------------------------------------
e_train.extend(train_e_idL[(99*83)+4:])
v_train.extend(train_v_idL[(99*83)+4:])
p_train.extend(train_p_idL[(99*83)+4:])

e_ground.extend(train_e_idH[(99*83)+4:])
v_ground.extend(train_v_idH[(99*83)+4:])
p_ground.extend(train_p_idH[(99*83)+4:])

# -----------------------------------------------------------------------------
# Split Data {Test}
# Test
indexs = [2, 19, 22, 39, 42, 59, 62, 79, 82]
et_train = []
vt_train = []
pt_train = []
et_ground = []
vt_ground = []
pt_ground = []

for index in indexs:

  temp0 = train_e_idL[(index-1)*83:index*83]
  et_train.extend(temp0)
  temp1 = train_v_idL[(index-1)*83:index*83]
  vt_train.extend(temp1)
  temp2 = train_p_idL[(index-1)*83:index*83]
  pt_train.extend(temp2)

  temp3 = train_e_idH[(index-1)*83:index*83]
  et_ground.extend(temp3)
  temp4 = train_v_idH[(index-1)*83:index*83]
  vt_ground.extend(temp4)
  temp5 = train_p_idH[(index-1)*83:index*83]
  pt_ground.extend(temp5)

et_train.extend(train_e_idL[98*83:(99*83)+4])
vt_train.extend(train_v_idL[98*83:(99*83)+4])
pt_train.extend(train_p_idL[98*83:(99*83)+4])

et_ground.extend(train_e_idH[98*83:(99*83)+4])
vt_ground.extend(train_v_idH[98*83:(99*83)+4])
pt_ground.extend(train_p_idH[98*83:(99*83)+4])

# -----------------------------------------------------------------------------
dat_length = len(e_train)
division = 2
train_parts = dat_length // division
test_parts = len(et_train) // 6
total_len = 0

for i in range(0,division):
  temp = len(e_train[i*train_parts:train_parts*(i+1)])
  total_len += temp

# -----------------------------------------------------------------------------
# Write e_train tfrecords
for i in range(0, division):
  with tf.io.TFRecordWriter('Train3_tfrecords/e_train/'\
                            'e_train_{}_.tfrecords'.format(i+1)) as writer:
    for image in e_train[train_parts*i:train_parts*(i+1)]:
        image = Image.open(TRAIN_PATH_eL + image)
        tf_example = img_example(image)
        writer.write(tf_example.SerializeToString())

# -----------------------------------------------------------------------------
# Write e_test (inputs) tfrecords
for i in range(0, 6):
  with tf.io.TFRecordWriter('Test2_tfrecords/e_train/'\
                            'e_train_{}_.tfrecords'.format(i+1)) as writer:
    for image in et_train[test_parts*i:test_parts*(i+1)]:
        image = Image.open(TRAIN_PATH_eL + image)
        tf_example = img_example(image)
        writer.write(tf_example.SerializeToString())

# ----------------------------------------------------------------------------- 
# Write v_train tfrecords
for i in range(0, division):
  with tf.io.TFRecordWriter('Train3_tfrecords/v_train/'\
                            'v_train_{}_.tfrecords'.format(i+1)) as writer:
    for image in v_train[train_parts*i:train_parts*(i+1)]:
        image = Image.open(TRAIN_PATH_vL + image)
        tf_example = img_example(image)
        writer.write(tf_example.SerializeToString())

# ----------------------------------------------------------------------------- 
# Write v_test (inputs) tfrecords
for i in range(0, 6):
  with tf.io.TFRecordWriter('Test2_tfrecords/v_train/'\
                            'v_train_{}_.tfrecords'.format(i+1)) as writer:
    for image in vt_train[test_parts*i:test_parts*(i+1)]:
        image = Image.open(TRAIN_PATH_vL + image)
        tf_example = img_example(image)
        writer.write(tf_example.SerializeToString())

# ----------------------------------------------------------------------------- 
# Write p_train tfrecords
for i in range(0, division):
  with tf.io.TFRecordWriter('Train3_tfrecords/p_train/'\
                            'p_train_{}_.tfrecords'.format(i+1)) as writer:
    for image in p_train[train_parts*i:train_parts*(i+1)]:
        image = Image.open(TRAIN_PATH_pL + image)
        tf_example = img_example(image)
        writer.write(tf_example.SerializeToString())

# -----------------------------------------------------------------------------  
# Write p_test (inputs) tfrecords
for i in range(0, 6):
  with tf.io.TFRecordWriter('Test2_tfrecords/p_train/'\
                            'p_train_{}_.tfrecords'.format(i+1)) as writer:
    for image in pt_train[test_parts*i:test_parts*(i+1)]:
        image = Image.open(TRAIN_PATH_pL + image)
        tf_example = img_example(image)
        writer.write(tf_example.SerializeToString())

# ----------------------------------------------------------------------------- 
# Write e_ground tfrecords
for i in range(0, division):
  with tf.io.TFRecordWriter('Train3_tfrecords/e_ground/'\
                            'e_ground_{}_.tfrecords'.format(i+1)) as writer:
    for image in e_ground[train_parts*i:train_parts*(i+1)]:
        image = Image.open(TRAIN_PATH_eH + image)
        tf_example = img_example(image)
        writer.write(tf_example.SerializeToString())

# ----------------------------------------------------------------------------- 
# Write e_test (outputs) tfrecords
for i in range(0, 6):
  with tf.io.TFRecordWriter('Test2_tfrecords/e_ground/'\
                            'e_ground_{}_.tfrecords'.format(i+1)) as writer:
    for image in et_ground[test_parts*i:test_parts*(i+1)]:
        image = Image.open(TRAIN_PATH_eH + image)
        tf_example = img_example(image)
        writer.write(tf_example.SerializeToString())

# ----------------------------------------------------------------------------- 
# Write v_ground tfrecords
for i in range(0, division):
  with tf.io.TFRecordWriter('Train3_tfrecords/v_ground/'\
                            'v_ground_{}_.tfrecords'.format(i+1)) as writer:
    for image in v_ground[train_parts*i:train_parts*(i+1)]:
        image = Image.open(TRAIN_PATH_vH + image)
        tf_example = img_example(image)
        writer.write(tf_example.SerializeToString())

# -----------------------------------------------------------------------------    
# # Write v_test (outputs) tfrecords
for i in range(0, 6):
  with tf.io.TFRecordWriter('Test2_tfrecords/v_ground/'\
                            'v_ground_{}_.tfrecords'.format(i+1)) as writer:
    for image in vt_ground[test_parts*i:test_parts*(i+1)]:
        image = Image.open(TRAIN_PATH_vH + image)
        tf_example = img_example(image)
        writer.write(tf_example.SerializeToString())

# ----------------------------------------------------------------------------- 
# Write p_ground tfrecords
for i in range(0, division):
  with tf.io.TFRecordWriter('Train3_tfrecords/p_ground/'\
                            'p_ground_{}_.tfrecords'.format(i+1)) as writer:
    for image in p_ground[train_parts*i:train_parts*(i+1)]:
        image = Image.open(TRAIN_PATH_pH + image)
        tf_example = img_example(image)
        writer.write(tf_example.SerializeToString())

# ----------------------------------------------------------------------------- 
# Write p_test (outputs) tfrecords
for i in range(0, 6):
  with tf.io.TFRecordWriter('Test_tfrecords/p_ground/'\
                            'p_ground_{}_.tfrecords'.format(i+1)) as writer:
    for image in pt_ground[test_parts*i:test_parts*(i+1)]:
        image = Image.open(TRAIN_PATH_pH + image)
        tf_example = img_example(image)
        writer.write(tf_example.SerializeToString())


# # Send Test_data to GCS
bucket_name_tfrecords = "ms_cfd_images/"
!gsutil -m cp -r /content/Test2_tfrecords gs://{bucket_name_tfrecords}

# Send Train_data to GCS
!gsutil -m cp -r /content/Train3_tfrecords gs://{bucket_name_tfrecords}
