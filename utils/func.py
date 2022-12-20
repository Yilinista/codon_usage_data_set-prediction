from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


def acc_check(model, pred, x_test, y_test):
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='micro')
    recall = recall_score(y_test, pred, average='micro')
    f1 = f1_score(y_test, pred, average='micro')
    auc = roc_auc_score(y_test, model.predict_proba(x_test), multi_class='ovr')
        
    print('\naccuracy_score: ', accuracy)
    print('precision_score: ', precision)
    print('recall_score: ',recall)
    print('f1_score: ', f1)
    print('auc:', auc)
    
    
def prediction(model, x_train, y_train, x_test):    
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
     
    return pred, model


def confusion_maxtrix(pred, y_test):    
    print('\nConfusion matrix :\n {0}\n'.format(confusion_matrix(y_test, pred)))
    print('Classification report :\n {0}'.format(classification_report(y_test, pred)))
    
    
def auc_plot(model, n_class, x_train, y_train, x_test, y_test):       
    y_score = model.fit(x_train, y_train).predict_proba(x_test)
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    y_onehot_test.shape  
        
    fig, ax = plt.subplots(figsize=(6, 6))
        
    if n_class == 3:
        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        class_name = ['nuclear', 'mitochondrion', 'chloroplast']
    elif n_class == 5:
        colors = cycle(["aqua", "darkorange", "cornflowerblue", 'darkgreen', 'red'])
        class_name = [ 'archaea', 'bacteria', 'eukaryote', 'bacteriophage', 'virus']

    for class_id, color in zip(range(n_class), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {class_name[class_id]}",
            color=color,
            ax=ax,
            )
           
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    

def nn_acc_check(model, pred, x_test, y_test):
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='micro')
    recall = recall_score(y_test, pred, average='micro')
    f1 = f1_score(y_test, pred, average='micro')
        
    print('\naccuracy_score: ', accuracy)
    print('precision_score: ', precision)
    print('recall_score: ',recall)
    print('f1_score: ', f1)    
    
    
def nn_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Acc.')
    plt.plot(val_acc, label='Validation Acc.')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')