""""
@author: jflack
"""

"""Importing Packages"""
#-------------------------------------------------------------------
import nibabel as nib #Allows access of NIFTI files
import os #Imports operating system to allow file access
from scipy import ndimage #Allows multidimensional image processing
import numpy as np #Importing numpy
import random #Import random module

#Importing tensorflow
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers
#-------------------------------------------------------------------



"""Image processing"""
#-------------------------------------------------------------------

def Image_Processor(path):
    #Desired dimensions of the final processed scan
    depth_final = 64
    width_final = 64
    height_final = 64
    
    #Read in file and get data using nibabel
    MRI_scan = nib.load(path)
    MRI_data = MRI_scan.get_fdata()
    
    # Get current depth
    depth_0 = MRI_data.shape[0]
    width_0 = MRI_data.shape[1]
    height_0 = MRI_data.shape[2]
    
    # Compute depth factor to scale image to (64,64,64)
    df = 1 / (depth_0 / depth_final)
    wf = 1 / (width_0 / width_final)
    hf = 1 / (height_0 / height_final)
    
    # Resize using multidemsnsional processing 
    MRI_resize = ndimage.zoom(MRI_data, (df, wf, hf,1), order=1)

    #Z-score normalisation of the MRI
    mean=np.mean(MRI_resize)
    stddev=np.std(MRI_resize)
    MRI_z=(MRI_resize-mean)/stddev
    
    #Correct MRI data to not be outside the minimum of maximum
    maximum=np.max(MRI_z)
    minimum=np.min(MRI_z)
    
    #Scale MRI data to between the set maximum and minimum values
    MRI_norm = (MRI_z - minimum) / (maximum - minimum)
    MRI_norm = MRI_norm.astype("float32")
    
    #Return resized and normalised image
    return MRI_norm

#Directories for data
ADdir_train='C:/Users/jflac/Documents/CNN/AD'
CNdir_train='C:/Users/jflac/Documents/CNN/CN'

#Making lists of the files in the CN and AD categories
CN_paths = [os.path.join(CNdir_train, i) for i in os.listdir(CNdir_train)]
AD_paths = [os.path.join(ADdir_train, i) for i in os.listdir(ADdir_train)]
    
print("Cognitively Normal Scans for training: " + str(len(CN_paths)))
print("Alzheimer's Scans for training: " + str(len(AD_paths)))

#Processing the scans using the function above
AD_scans = np.array([Image_Processor(path) for path in AD_paths])
CN_scans = np.array([Image_Processor(path) for path in CN_paths])

#Shuffling data set before split into training and testing
random.shuffle(AD_scans)
random.shuffle(AD_scans)

#Creating arrays of labels for the scans
AD_labels = np.array([1 for _ in range(len(AD_scans))])
CN_labels = np.array([0 for _ in range(len(CN_scans))])

#Combining the CN and AD data into a training set of size 170
x_train = np.concatenate((AD_scans[:85], CN_scans[:85]), axis=0)
y_train = np.concatenate((AD_labels[:85], CN_labels[:85]), axis=0)

#Combining the CN and AD data into a testing set of size 65
x_test = np.concatenate((AD_scans[:85], CN_scans[:85]), axis=0)
y_test = np.concatenate((AD_labels[:85], CN_labels[:85]), axis=0)

#-------------------------------------------------------------------------

"""Model Setup"""
#-------------------------------------------------------------------------
# Define training data loader
training_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))

#Code to rotate the MRI's semi-randomly
def rotator(volume):
    #Remove 1 from (depth,witdth,height,1)
    volume=tf.squeeze(volume)
    #Define a series of angles to be chosen randomly for each image each epoch
    turn = [-20,-15,-10,-5,5,10,15,20]
    turn = random.choice(turn)
    #Rotate the volume
    volume = ndimage.rotate(volume, turn, reshape=False)
    #Capping values so dont exceed 0 and 1
    volume[volume < 0] = 0
    volume[volume > 1] = 1
    #Add 1 from (depth,witdth,height,1)
    volume=tf.expand_dims(volume,axis=3)
    return volume

#Function for the dataset 
def training_rotator(volume, label):
    # Rotate volume and return as a tensor
    volume = tf.numpy_function(rotator, [volume], tf.float32)
    return volume, label  
    
training_dataset = (
    training_loader.shuffle(len(x_train)) #Shuffle over the length of the set
    .map(training_rotator) #Change the angles
    .batch(5) #Analyse in batches of 5
    .prefetch(1) #Preload 1
)
#-----------------------------------------------------------------------

"""Build a 3D convolutional neural network model"""
#-----------------------------------------------------------------------
#Build a model with input 64, 64, 64
def model_maker(width=64, height=64, depth=64):
    
    #Input in the shape (64,64,64,1)
    inputs = keras.Input((width, height, depth, 1))
    
    #Creating a convolutional layer with 32 filters
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    
    #Max pooling with size 2 to downsample
    x = layers.MaxPool3D(pool_size=2)(x)
    
    #Recentreing and rescaling the inputs 
    x = layers.BatchNormalization()(x)

    #Creating a convolutional layer with 64 filters
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    
    #Max pooling with size 2 to downsample
    x = layers.MaxPool3D(pool_size=2)(x)
    
    #Recentreing and rescaling the inputs 
    x = layers.BatchNormalization()(x)

    #Creating a convolutional layer with 128 filters
    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    
    #Max pooling with size 2 to downsample
    x = layers.MaxPool3D(pool_size=2)(x)
    
    #Recentreing and rescaling the inputs 
    x = layers.BatchNormalization()(x)

    #Creating a convolutional layer with 256 filters
    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    
    #Max pooling with size 2 to downsample
    x = layers.MaxPool3D(pool_size=2)(x)
    
    #Recentreing and rescaling the inputs 
    x = layers.BatchNormalization()(x)
    
    #Computes a global average of each input
    x = layers.GlobalAveragePooling3D()(x)
    
    #Fully connected dense layer
    x = layers.Dense(units=512, activation="relu")(x)
    
    #Recentreing and rescaling the inputs 
    x = layers.BatchNormalization()(x)
    
    #Dropout of 0.4
    x = layers.Dropout(0.4)(x)
    
    #Calculating the output as a value between 1 and 0
    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="tbd-net")
    return model


#Build model with height width and depth of 64
model = model_maker(width=64, height=64, depth=64)
model.summary()
#-----------------------------------------------------------------------

"""Model Fitting"""
#-----------------------------------------------------------------------
#Compile model with initial learning rate 0.00001 and exponential decay
initial_learning_rate = 0.00001
#decay steps set to 1000 with decay rate of 0.96
lr = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=1000, decay_rate=0.96)

#Compiling the model with loss function of binary crossentropy and an Adam optimizer with metric of accuracy
model.compile(loss="binary_crossentropy",metrics=['accuracy'],optimizer=keras.optimizers.Adam(learning_rate=lr))

# Train the model for 100 epochs with shuffle on and verbose of 2 for feedback bars each epoch
epochs = 100
data=model.fit(
    training_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
)