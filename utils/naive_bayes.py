import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

from utils.func import acc_check, prediction, confusion_maxtrix, auc_plot

class naive_bayes():
    def __init__(self, x_train, y_train, x_test, y_test, default_params, cv_params, fold_cv=5):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.default_params = default_params
        self.cv_params = cv_params
        self.fold_cv = fold_cv
        
        self.n_class = len(np.unique(y_train))
        
        
    def cross_validation(self, default_params, cv_params, mode):
        x_train = self.x_train
        y_train = self.y_train
        fold_cv = self.fold_cv
        
        if mode == 'bernoulli':
            model = BernoulliNB(**default_params)
        elif mode == 'gaussian':
            model = GaussianNB(**default_params)
        elif mode == 'complement':
            model = ComplementNB(**default_params)
        elif mode == 'multinomial':
            model = MultinomialNB(**default_params)
            
        cv = GridSearchCV(model, cv_params, verbose=2, refit=True, cv=fold_cv, n_jobs=-1)
        cv.fit(x_train,y_train) 
        
        print('Best params: ', cv.best_params_)
        print('Best scores: ', cv.best_score_)
        
        self.best_params_ = cv.best_params_
        
        return cv.best_params_, model  

 
    def generate(self, mode):
        self.mode = mode
        self.best_params_, self.model = self.cross_validation(self.default_params, self.cv_params, self.mode)
        
        if mode == 'bernoulli':
            self.model = BernoulliNB(**self.default_params, **self.best_params_)
        elif mode == 'gaussian':
            self.model = GaussianNB(**self.default_params, **self.best_params_)
        elif mode == 'complement':
            self.model = ComplementNB(**self.default_params, **self.best_params_)
        elif mode == 'multinomial':
            self.model = MultinomialNB(**self.default_params, **self.best_params_)
            
        self.pred, self.model = prediction(self.model, self.x_train, self.y_train, self.x_test)
        acc_check(self.model, self.pred, self.x_test, self.y_test)        
        confusion_maxtrix(self.pred, self.y_test)
        auc_plot(self.model, self.n_class, self.x_train, self.y_train, self.x_test, self.y_test)
        