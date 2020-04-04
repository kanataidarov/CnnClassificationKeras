from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf 
import os 
import numpy as np 
from matplotlib import pyplot as plt 

if not os.path.isdir('models'): 
    os.mkdir('models')

print('Tensorflow version: ', tf.__version__)
# print('Is using GPU? ', tf.test.is_gpu_available())

# Restrict datasets 
from tensorflow.keras.utils import to_categorical

def get_three_classes(x, y):
    def indices_of(class_id):
        indices, _ = np.where(y == float(class_id))
        return indices

    indices = np.concatenate([indices_of(0), indices_of(1), indices_of(2)], axis=0)
    
    x = x[indices]
    y = y[indices]
    
    count = x.shape[0]
    indices = np.random.choice(range(count), count, replace=False)
    
    x = x[indices]
    y = y[indices]
    
    y = tf.keras.utils.to_categorical(y)
    
    return x, y

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train, y_train = get_three_classes(x_train, y_train)
x_test, y_test = get_three_classes(x_test, y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Visialize initial datasets 
class_names = ['aeroplane', 'car', 'bird']

def show_random_examples(x, y, pred): 
    n_examples = 10
    indices = np.random.choice(range(x.shape[0]), n_examples, replace=False)

    x = x[indices]
    y = y[indices]
    pred = pred[indices]

    plt.figure(figsize=(10, 5))
    ims_per_row = 5
    for i in range(n_examples): 
        plt.subplot(n_examples/ims_per_row, ims_per_row, i+1)
        plt.imshow(x[i])
        plt.xticks([])
        plt.yticks([])
        clr = 'green' if np.argmax(y[i]) == np.argmax(pred[i]) else 'red'
        plt.xlabel(class_names[np.argmax(pred[i])], color=clr)
    plt.show()

# show_random_examples(x_train, y_train, y_train)
# show_random_examples(x_test, y_test, y_test)

# Building model 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Input, Dense

def create_model():
    
    def add_conv_block(model, num_filters):
        
        model.add(Conv2D(num_filters, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(num_filters, 3, activation='relu', padding='valid'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))

        return model
    
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(32, 32, 3)))
    
    model = add_conv_block(model, 32)
    model = add_conv_block(model, 64)
    model = add_conv_block(model, 128)

    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()
model.summary()

# Traning model 
epochs = 10
train_history = model.fit(
    x=x_train/255., y=y_train,
    validation_data=(x_test/255., y_test),
    epochs=epochs, batch_size=128,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2),
        tf.keras.callbacks.ModelCheckpoint(monitor='val_accuracy', save_best_only=True, save_weights_only=False, filepath='models/model_1.h5')
    ]
)

# Predictions
accs = train_history.history['accuracy']
val_accs = train_history.history['val_accuracy']

plt.plot(range(len(accs), accs, label='Trainging accs'))
plt.plot(range(len(val_accs), val_accs, label='Validation accs'))
plt.legend()
plt.show()

model = tf.keras.models.load_model('models/model_1.h5')
preds = model.predict(x_test/255.)
show_random_examples(x_test, y_test, preds)

