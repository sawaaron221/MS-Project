"""
Aaron Tun
MS Project 
"""
import tensorflow as tf
# -----------------------------------------------------------------------------
def ConvBlock(inputs, cnn_filter, kernel_grid, dropout=0.1, alpha=0.1, 
                cnn_type ='Block Type', inputs2='none', inputs3='none'):
    """
    Return a Convolutional Block. 
    Args:
        inputs      (float)  - Image size 256 x 256, 3 channels (RGB).
        inputs2     (float)  - Image size 256 x 256, 3 channels (RGB).
        inputs3     (float)  - Image size 256 x 256, 3 channels (RGB).
        cnn_filter  (int)    - Number of filters for cnn layer.
        kernel_gird (int)    - Number of grid (3 x 3).
        dropout     (float)  - Dropout value: 0.0 to 1.0.
        alpha       (float)  - alpha value for LeakyRelu: 0.0 to 1.0.
        cnn_type    (string) - Type of cnn block eg. contraction, central, 
                               expansion block.
        
    Returns:
        c           (float)  - Return a tensor for next convolution block.
    """
# -----------------------------------------------------------------------------
    # Contraction Convolutional Neural Network Block
    if cnn_type == 'Contraction':
        c = tf.keras.layers.Conv2D(cnn_filter, (kernel_grid, kernel_grid), 
                                   activation=tf.keras.layers.LeakyReLU(alpha), 
                                   kernel_initializer='he_normal', 
                                   padding='same')(inputs)
        c = tf.keras.layers.BatchNormalization(axis=-1)(c)
        c = tf.keras.layers.Dropout(dropout)(c)
        c = tf.keras.layers.Conv2D(cnn_filter, (kernel_grid, kernel_grid), 
                                   activation=tf.keras.layers.LeakyReLU(alpha), 
                                   kernel_initializer='he_normal', 
                                   padding='same')(c)
        p = tf.keras.layers.MaxPooling2D((2,2))(c)
        return p, c
# -----------------------------------------------------------------------------
    # Expansion Convolutional Neural Network Block
    elif cnn_type == 'Expansion':
        u = tf.keras.layers.Conv2DTranspose(cnn_filter, (2,2), strides=(2,2), 
                                            padding='same')(inputs)
        u = tf.keras.layers.concatenate([u,inputs2])
        c = tf.keras.layers.Conv2D(cnn_filter, (kernel_grid, kernel_grid), 
                                   activation='relu', 
                                   kernel_initializer='he_normal', 
                                   padding='same')(u)
        c = tf.keras.layers.BatchNormalization(axis=-1)(c)  
        c = tf.keras.layers.Dropout(dropout)(c)
        c = tf.keras.layers.Conv2D(cnn_filter, (kernel_grid, kernel_grid), 
                                   activation='relu', 
                                   kernel_initializer='he_normal', 
                                   padding='same')(c)
        return c
# -----------------------------------------------------------------------------
    # Concatenation Layer
    elif cnn_type == 'Concatenate':
        u = tf.keras.layers.concatenate([inputs,inputs2,inputs3])
        return u
# -----------------------------------------------------------------------------
    # Center Convolutional Neural Network Block
    elif cnn_type == 'CenterBlock':
        c = tf.keras.layers.Conv2D(cnn_filter, (kernel_grid, kernel_grid), 
                                   activation=tf.keras.layers.LeakyReLU(alpha), 
                                   kernel_initializer='he_normal', 
                                   padding='same')(inputs)
        c = tf.keras.layers.BatchNormalization(axis=-1)(c)
        c = tf.keras.layers.Dropout(dropout)(c)
        c = tf.keras.layers.Conv2D(cnn_filter, (kernel_grid, kernel_grid), 
                                   activation=tf.keras.layers.LeakyReLU(alpha), 
                                   kernel_initializer='he_normal', 
                                   padding='same')(c)
        return c
# -----------------------------------------------------------------------------    
    else:
        print('Invalid Inputs')
# -----------------------------------------------------------------------------
