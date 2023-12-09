from pathlib import Path
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from keras.callbacks import EarlyStopping
from random import randrange
from keras.models import load_model

class CNNModel:

    def load_images(self):
        image_dir = Path('dataset')
        images_path_list = list(image_dir.glob('*.jpg'))

        # get file names which act as english word for braille image
        self.__names_list = [image_path.name[0]
                             for image_path in images_path_list]

        # convert images into arrays of images
        images = [cv2.imread(str(dir)) for dir in images_path_list]
        # convert images to 0's and 1's
        self.__images_list = np.array(images) / 255.0
        self.__names_list = np.array(self.__names_list)

    def prepare_data(self):
        # convert each alphabet character to numerical value. for example a,b,c to 0,1,2
        label_encoder = LabelEncoder()
        # numerical values
        encoded_labels = label_encoder.fit_transform(self.__names_list)
    
        # split data into train set for training the model and test set for testing the model
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(
            self.__images_list, encoded_labels, test_size=0.2, random_state=42)
        
        self.__classes_names = label_encoder.classes_
            
    def build_model_layers(self):
        self.__model = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(
                1, 1), activation='relu', input_shape=(28, 28, 3)),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(filters=128, kernel_size=(
                3, 3), padding='same', strides=(1, 1), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.2, input_shape=(28, 1)),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(filters=256, kernel_size=(
                3, 3), padding='same', strides=(1, 1), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25, input_shape=(28, 1)),
            keras.layers.BatchNormalization(),

            keras.layers.Flatten(),

            keras.layers.Dense(units=512, activation="relu"),
            keras.layers.Dropout(0.5, input_shape=(28, 1)),
            keras.layers.BatchNormalization(),

            keras.layers.Dense(units=288, activation="relu"),

            # units=26 mean number of output classes. Each alphabet is one unit or one class
            keras.layers.Dense(units=26, activation="softmax")

        ])

    def train_model(self):
        # path to save and retrieve model
        MODEL_PATH = 'trained_model/cnn_model.h5'

        # model already exists
        if os.path.exists(MODEL_PATH):
            # return with already existing model
            self.__model = load_model(MODEL_PATH)
            return

        # path to save model weights after every 5*32 = 160 batch size
        MODEL_CHECKPOINT_TEMP_PATH = "model_checkpoints_temp/braille.ckpt"
        batch_size = 32

        # callback that save weights after every 5*32 = 160 batch size
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_CHECKPOINT_TEMP_PATH,
            verbose=1,
            save_weights_only=True,
            save_freq=5*batch_size)

        # compile model with configuration
        self.__model.compile(optimizer="Adam", loss="SparseCategoricalCrossentropy",
                      metrics=["sparse_categorical_accuracy"])

        # stop training process if there is no improvement in the monitored metrics for 20 consecutive epochs.
        early_stopping_val_sparse_categorical_accuracy = EarlyStopping(
            patience=20, monitor="val_sparse_categorical_accuracy", mode="auto")
        early_stopping_val_loss = EarlyStopping(
            patience=20, monitor="val_loss", mode="auto")

        # train model
        self.__training_history = self.__model.fit(x=self.__x_train,
                                     y=self.__y_train,
                                     epochs=100,
                                     validation_split=0.3,
                                     callbacks=[early_stopping_val_sparse_categorical_accuracy, early_stopping_val_loss, save_callback])

        # model.summary()

        # save model in file
        self.__model.save(MODEL_PATH)

    def evaluate_model(self):
        if(self.__model is None):
            print("No model to evaluate")
            return
        
        # evaluate trained model using test data
        test_loss, test_acc = self.__model.evaluate(self.__x_test, self.__y_test)
        print('Loss', test_loss)
        print('Accuracy', test_acc)

    def test_model(self):
       # predict one random image
        index_image = randrange(312)
        prediction_scores = self.__model.predict(
            np.expand_dims(self.__x_test[index_image], axis=0))

        # get most matched alphabet index
        predicted_index = np.argmax(prediction_scores)

        #show image
        plt.imshow(self.__x_test[index_image])
        plt.show()

        #print predictions
        print("score: ", prediction_scores)
        print("Predicted label: " + self.__classes_names[predicted_index])

    def plot_training_history(self):
        if (self.__training_history is None):
            print("No model training is performed. Cannot plot model training history")
            return

        time = np.arange(1, len(self.__training_history.history['loss'])+1)

        # Loss Fitting History Plot
        sns.lineplot(data=self.__training_history.history, x=time, y='loss')
        sns.lineplot(data=self.__training_history.history, x=time, y='val_loss')
        plt.title('Loss fitting history')
        plt.legend(labels=['Loss', 'Validation Loss'])
        plt.show()

        # Accuracy Fitting History Plot
        sns.lineplot(data=self.__training_history.history, x=time,
                     y='val_sparse_categorical_accuracy')
        sns.lineplot(data=self.__training_history.history, x=time,
                     y='sparse_categorical_accuracy')
        plt.title('Accuracy fitting history')
        plt.legend(labels=['Accuracy', 'Validation Accuracy'])
        plt.show()
        
    def plot_30_train_images_with_labels(self):
        # class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 
        #             'h', 'i', 'j', 'k', 'l', 'm', 'n', 
        #             'o', 'p', 'q', 'r', 's', 't', 'u', 
        #             'v', 'w', 'x', 'y', 'z']

        plt.figure(figsize=(10,10))
        for i in range(30):
            plt.subplot(6,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.__x_train[i])
            plt.xlabel(self.__names_list[self.__y_train[i]])
        plt.show()
