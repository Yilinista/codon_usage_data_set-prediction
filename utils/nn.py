import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV

from utils.func import prediction, confusion_maxtrix, nn_acc_check, nn_plot

class nn():
    def __init__(self, x_train, y_train, x_test, y_test, layers, fold_cv=5):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.layers = layers
        self.fold_cv = fold_cv
        self.n_class = len(np.unique(y_train))
        
        self.n_class = len(np.unique(y_train))
        
    def train(self, layers, x_train, y_train, x_test, y_test, n_class):
        if len(layers) == 1:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(layers[0], activation='relu'), 
                tf.keras.layers.Dense(n_class, activation='softmax')
                ])
        elif len(layers) == 2:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(layers[0], activation='relu'), 
                tf.keras.layers.Dense(layers[1], activation='relu'), 
                tf.keras.layers.Dense(n_class, activation='softmax')
                ])
            
        model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])     
        
        history = model.fit(x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test, y_test)) 
        
        return model, history
 

    def prediction(self, model, x_test):
        y_pred = model.predict(x_test)
        y_pred1 = np.zeros(y_pred.shape[0])
        for index, value in enumerate(y_pred):
            y_pred1[index] = np.argmax(y_pred[index])
            
        return y_pred1


    def generate(self):     
        self.model, self.history = self.train(self.layers, self.x_train, self.y_train, self.x_test, self.y_test, self.n_class)
        self.pred = self.prediction(self.model, self.x_test)
        nn_acc_check(self.model, self.pred, self.x_test, self.y_test)        
        confusion_maxtrix(self.pred, self.y_test)
        nn_plot(self.history)
        
        