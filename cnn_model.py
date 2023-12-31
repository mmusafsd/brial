from pathlib import Path
import numpy as np
from keras import regularizers
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

    def __init__(self):
        self._KAGGLE_DATASET_PATH = "dataset/kaggle_dataset"
        self._TEST_DATASET_PATH = "dataset/test_dataset"
        self._MODEL_CHECKPOINT_TEMP_PATH = "model_checkpoints_temp/braille.ckpt"
        self._MODEL_PATH = 'trained_model/cnn_model.h5'
        self._model = None
        self._label_class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
                                   'h', 'i', 'j', 'k', 'l', 'm', 'n',
                                   'o', 'p', 'q', 'r', 's', 't', 'u',
                                   'v', 'w', 'x', 'y', 'z']

    def load_model(self):
        self._load_images()
        self._prepare_data()
        self._load_model()

    def _load_model(self):
        if (os.path.exists(self._MODEL_PATH)):
            # already existing model
            self._model = load_model(self._MODEL_PATH)

    def _load_images(self):
        image_dir = Path(self._KAGGLE_DATASET_PATH)
        images_path_list = list(image_dir.glob('*.jpg'))

        # get file names which act as english word for braille image
        self._labels_list = [image_path.name[0]
                             for image_path in images_path_list]

        # convert colorful images into arrays of images
        self._images = [cv2.imread(str(dir)) for dir in images_path_list]

    def _prepare_data(self):
        # match image size and shape with model input size and shape required
        resize_images = []
        for image in self._images:
            # convert color image to gray scale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # resize image to fit model input size
            resize_images.append(cv2.resize(image, (28, 28)))

        # convert images to 0's and 1's
        self._images_list = np.array(resize_images) / 255.0
                
        # convert each alphabet character to numerical value. for example a,b,c to 0,1,2
        label_encoder = LabelEncoder()
        # numerical values
        encoded_labels = label_encoder.fit_transform(self._labels_list)

        # split data into train set for training the model and test set for testing the model
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(
            self._images_list, encoded_labels, test_size=0.2, random_state=42)

    def _build_model_layers(self):
        self._model = keras.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1),kernel_regularizer=regularizers.l2()),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2()),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.BatchNormalization(),

            keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2()),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.BatchNormalization(),

            keras.layers.Flatten(),

            keras.layers.Dense(units=128, activation="relu", kernel_regularizer=regularizers.l2()),
            keras.layers.Dropout(0.5),
            
            # units=26 mean number of output classes. Each alphabet is one unit or one class
            keras.layers.Dense(units=26, activation="softmax", kernel_regularizer=regularizers.l2())

        ])

    def _build_model(self):
        # path to save model weights after every 5*32 = 160 batch size
        batch_size = 32

        # callback that save weights after every 5*32 = 160 batch size
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self._MODEL_CHECKPOINT_TEMP_PATH,
            verbose=1,
            save_weights_only=True,
            save_freq=5*batch_size)

        # compile model with configuration
        self._model.compile(optimizer="adam", loss="SparseCategoricalCrossentropy",
                            metrics=["sparse_categorical_accuracy"])

        # stop training process if there is no improvement in the monitored metrics for 20 consecutive epochs.
        early_stopping_val_sparse_categorical_accuracy = EarlyStopping(
            patience=20, monitor="val_sparse_categorical_accuracy", mode="auto")
        early_stopping_val_loss = EarlyStopping(
            patience=20, monitor="val_loss", mode="auto")

        # train model
        self.__training_history = self._model.fit(x=self.__x_train,
                                                  y=self.__y_train,
                                                  epochs=100,
                                                  validation_split=0.3,
                                                  callbacks=[early_stopping_val_sparse_categorical_accuracy, early_stopping_val_loss, save_callback])

        # model.summary()

        # save model in file
        self._model.save(self._MODEL_PATH)

    def train_model(self):
        self._load_images()
        self._prepare_data()
        self._build_model_layers()
        self._build_model()

    def evaluate_model(self):
        if (self._model is None):
            print("No model to evaluate")
            return

        # evaluate trained model using test data
        test_loss, test_acc = self._model.evaluate(
            self.__x_test, self.__y_test)
        print('Loss', test_loss)
        print('Accuracy', test_acc)

    def test_model(self):
        if (self._model is None):
            print("No model is found.")
            return

        image_dir = Path(self._TEST_DATASET_PATH)
        image_path_list = list(image_dir.glob('*'))

        total_predictions = len(image_path_list)
        total_no_of_correct_predictions = 0
        for image_path in image_path_list:
            # preprocess the image
            image = cv2.imread(str(image_path))

            # convert color image to gray scale
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # resize image to fit model
            image = cv2.resize(image, (28, 28))

            # convert images to 0's and 1's
            image = np.array(image) / 255.0

            original_label = image_path.name[0]

            prediction_scores = self._model.predict(
                np.expand_dims(image, axis=0))

            # get most matched alphabet index
            predicted_index = np.argmax(prediction_scores)

            # get predicated label
            predicted_label = self._label_class_names[predicted_index]

            # count the number of correct predictions
            if original_label == predicted_label:
                total_no_of_correct_predictions += 1
            # print predictions
            # print("score: ", prediction_scores)
            print("Original label: " + original_label)
            print("Predicted label: " + predicted_label)

        # calculate predictions success percentage
        total_correct_predications_percentage = (
            total_no_of_correct_predictions / total_predictions) * 100
        total_correct_predications_percentage = round(
            total_correct_predications_percentage, 2)

        # print total predications detail
        print("Total Predictions: ", total_predictions)
        print("Total Correct Predictions: ", total_no_of_correct_predictions)
        print("Total Correct Predictions Percentage(%): " +
              str(total_correct_predications_percentage)+"%")

        # show image
        # plt.imshow(image)
        # plt.show()

    def plot_training_history(self):
        if (self._model is None):
            print("No model is found.")
            return
        if (self.__training_history is None):
            print("No model training is performed. Cannot plot model training history")
            return

        time = np.arange(1, len(self.__training_history.history['loss'])+1)

        # Loss Fitting History Plot
        sns.lineplot(data=self.__training_history.history, x=time, y='loss')
        sns.lineplot(data=self.__training_history.history,
                     x=time, y='val_loss')
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

        plt.figure(figsize=(10, 10))
        for i in range(30):
            plt.subplot(6, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.__x_train[i])
            plt.xlabel(self._labels_list[self.__y_train[i]])
        plt.show()
