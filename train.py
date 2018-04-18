
# coding: utf-8

# In[18]:


import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
print(tf.__version__)
print(keras.__version__)


# In[13]:


import keras
from keras.preprocessing.image import ImageDataGenerator

# import necessary building blocks
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU


# In[7]:


train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True, validation_split=0.2)


# In[8]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[9]:


train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical', subset='training')


# In[10]:


validation_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical', subset='validation')


# In[27]:


def make_model():
    """
    Define your model architecture here.
    Returns `Sequential` model.
    """
    model = Sequential()

    ### YOUR CODE HERE
    model.add(Conv2D(16, (3,3),input_shape=(150, 150, 3), padding='same'))
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(256))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))
    
    # model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


# In[28]:


# describe model
K.clear_session()  # clear default graph
model = make_model()
model.summary()


# In[29]:


INIT_LR = 5e-3  # initial learning rate
BATCH_SIZE = 32
EPOCHS = 10

K.clear_session()  # clear default graph
# don't call K.set_learning_phase() !!! (otherwise will enable dropout in train/test simultaneously)
model = make_model()  # define our model

# prepare model for fitting (loss, optimizer, etc)
model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=keras.optimizers.adamax(lr=INIT_LR),  # for SGD
    metrics=['accuracy']  # report accuracy during training
)

# scheduler of learning rate (decay with epochs)
def lr_scheduler(epoch):
    return INIT_LR * 0.9 ** epoch

# callback for printing of actual learning rate used by optimizer
class LrHistory(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("Learning rate:", K.get_value(model.optimizer.lr))


# In[30]:


# fit model
# model.fit_generator(
#     x_train2, y_train2,  # prepared data
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler), 
#                LrHistory(), 
#                keras_utils.TqdmProgressCallback(),
#                keras_utils.ModelSaveCallback(model_filename)],
#     validation_data=(x_test2, y_test2),
#     shuffle=True,
#     verbose=0,
#     initial_epoch=last_finished_epoch or 0
# )


model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)


# In[ ]:


# save weights to file
model.save_weights("/artifacts/weights.h5")


# In[ ]:


# load weights from file (can call without model.fit)
model.load_weights("/artifacts/weights.h5")

