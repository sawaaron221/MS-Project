"""
Aaron Tun
MS Project 
"""
import os
IS_COLAB_BACKEND = 'COLAB_GPU' in os.environ  
if IS_COLAB_BACKEND:
  from google.colab import auth
  auth.authenticate_user()

# install tensorboard plugin
!pip install -U pip install -U tensorboard_plugin_profile

# Load the TensorBoard notebook extension and declare GCS bucket file path.
%load_ext tensorboard
bucket_name_data = "ms_cfd_images/Tensorboard_2"

# Remove Older Versions
!gsutil -m rm -r gs://{bucket_name_data}/

# Clear any logs from previous runs
!rm -rf ./Tensorboard/

# Download Tensorboard Files From GCS
# =============================================================================
!gsutil -m cp -r gs://{bucket_name_data} .
# =============================================================================

# Launch TensorBoard for Concatenated model.
%tensorboard --logdir /content/drive/MyDrive/Tensorboard/Concat_CNN_Tensorboard

# Launch TensorBoard for Split model.
%tensorboard --logdir /content/drive/MyDrive/Tensorboard/Split_CNN_Tensorboard

