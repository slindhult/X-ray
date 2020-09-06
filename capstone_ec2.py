import pandas as pd
import boto3.docs.attr
import numpy as np
from itertools import chain

#Clean Data

data = pd.read_csv('s3://galvanize-capstone/Data_Entry_2017.csv')
data.drop('Unnamed: 11', axis=1, inplace=True)

paths = pd.read_csv('s3://galvanize-capstone/paths.csv')
paths.drop('Unnamed: 0', axis=1, inplace=True)

data = data.join(paths)
data['path'] = data['path'].apply(lambda x: 'local_dir/' + x)

data = data[data['Patient Age']<120]

data['clean_labels'] = data['Finding Labels'].apply(lambda x: x.split('|'))
labels = data['clean_labels'].str.join('|').str.get_dummies()
data = data.join(labels)

data['Finding Labels'] = data['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

all_labels = labels.columns
clean_labels = []
for i in all_labels:
    j = data[i].sum()
    if j >1000 and i != 'No Finding':
        clean_labels.append(i)
    else:
        print(i)

data.drop(['Hernia', 'No Finding'], axis=1, inplace=True)

data['disease_vec'] = data.apply(lambda x: [x[clean_labels].values], 1).map(lambda x: x[0])

hern_dex = [i for i in data.index if 'Hernia' in data['Finding Labels'].loc[i]]

data = data[data['Finding Labels']!='']
data.drop(index=hern_dex, inplace=True)



#Modeling
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(data,
                                   test_size = 0.25)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator(samplewise_center=True,
                              samplewise_std_normalization=True,
                              height_shift_range= 0.05,
                              width_shift_range=0.1,
                              rotation_range=5,
                              shear_range = 0.1,
                              fill_mode = 'constant',
                              zoom_range=0.15)

image_size = (128, 128)

train_gen = img_gen.flow_from_dataframe(dataframe=train_df, x_col = 'path',
y_col = 'clean_labels', classmode = 'multi_output',
classes = clean_labels, targetsize = image_size, colormode = 'grayscale',
batch_size = 64)

test_gen = img_gen.flow_from_dataframe(dataframe=test_df, x_col = 'path',
y_col = 'clean_labels', classmode = 'multi_output',
classes = clean_labels, targetsize = image_size, colormode = 'grayscale',
batch_size = 256)

test_X, test_Y = next(img_gen.flow_from_dataframe(dataframe=test_df,
x_col = 'path', y_col = 'clean_labels',
classmode = 'multi_output', classes = clean_labels,
targetsize = image_size,
colormode = 'grayscale', batchsize = 1024))

from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense


# dimensions of our images.
img_width, img_height = 256, 256


from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_unflip_model_weights.best.hdf5".format('xray_class')

checkpoint3 = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=3)
callbacks_list3 = [checkpoint3, early]

model = Sequential()
model.add(Conv2D(128, (2, 2), input_shape=(256,256,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense((len(clean_labels))))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['categorical_accuracy'])

model.fit(train_gen,
                                  steps_per_epoch=int(test_df.shape[0]/64),
                                  validation_data = (test_X, test_Y),
                                  epochs = 10,
                                  callbacks = callbacks_list3)










