from __future__ import absolute_import, division, print_function
from keras.callbacks import ModelCheckpoint


# Import_Tenserflow
import tensorflow as tf


import numpy as np
import matplotlib.pyplot as plt


# Load_Fashion_Mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


# Visualize_the_data
print("x_train shape :", x_train.shape, "y_train shape : ", y_train.shape)

print(x_train.shape[0], 'train set')
print(y_train.shape[0], 'test set')


# define the text labels
fashion_mnist_lables = ["T-shirts/top", # index 0
                        "Trouser",      # index 1
                        "Pulloer",      # index 2
                        "Dress",        # index 3
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6
                        "Sneaker",      # index 7
                        "Bag",          # index 8
                        "Ankle boot"]   # index 9


img_index = 5
label_index = y_train[img_index]

# print_labels
print("y = " + str(label_index) + " " + (fashion_mnist_lables[label_index]))
plt.imshow(x_train[img_index])


# Data_normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print("Number of train data  -  " + str(len(x_train)))
print("Number of test data - " + str(len(x_test)))


# split_Data_into_Two_set
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]


# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')


# model
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))


# Detailed model construction
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss= 'categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)

model.fit(x_train,
          y_train,
          batch_size=64,
          epochs=10,
          validation_data=(x_valid, y_valid),
          #callbacks=[checkpointer]
          )

#model.load_weights('model.weights.best.hdf5')

score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])


y_hat = model.predict(x_test)

figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0],size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i+1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])

    ax.set_title("{} ({})".format(fashion_mnist_lables[predict_index],
                                  fashion_mnist_lables[true_index],
                                  color=("green" if predict_index == true_index else "red")))
