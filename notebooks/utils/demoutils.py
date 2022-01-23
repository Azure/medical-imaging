import time
import torch
import torch.nn.functional as F
import mlflow
import copy
from tqdm.autonotebook import tqdm
import numpy as np
from datetime import datetime

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'


def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=20, device='cuda'):
    '''
    Runs training loop for classification problems. Returns Keras-style
    per-epoch history of loss and accuracy over training and validation data.

    Parameters
    ----------
    model : nn.Module
        Neural network model
    optimizer : torch.optim.Optimizer
        Search space optimizer (e.g. Adam)
    loss_fn :
        Loss function (e.g. nn.CrossEntropyLoss())
    train_dl : 
        Iterable dataloader for training data.
    val_dl :
        Iterable dataloader for validation data.
    epochs : int
        Number of epochs to run
    device : string
        Specifies 'cuda' or 'cpu'

    Returns
    -------
    Dictionary
        Similar to Keras' fit(), the output dictionary contains per-epoch
        history of training loss, training accuracy, validation loss, and
        validation accuracy.
    '''

    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    
    start_time_sec = time.time()
    
    # initialize Hyperdrive metric
    best_val_acc = 0.0
    best_val_loss = 0.0
    
    for epoch in range(epochs):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0

        for batch in train_dl:

            optimizer.zero_grad()

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x.size(0)
            num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)


        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for batch in val_dl:

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x.size(0)
            num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        mlflow.log_metric('training loss', train_loss)
        mlflow.log_metric('validation loss', val_loss)
        mlflow.log_metric('training accuracy', train_acc)
        mlflow.log_metric('validation accuracy', val_acc)

        # check if this is the best epoch so far:
        is_best_epoch = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            is_best_epoch = True
            
            # save model state related to this epoch
            best_model_wts = copy.deepcopy(model.state_dict())
        
        print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f ' % \
               (epoch+1, epochs, train_loss, train_acc, val_loss, val_acc) + ('<- best epoch so far' if is_best_epoch else ''))
       
        
    # END OF TRAINING LOOP

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()

    return total_time_sec, time_per_epoch_sec, model, history

def plot_learning_curve(history):
    plt.rcParams.update(plt.rcParamsDefault)
    
    epochs = range(1, len(history['acc']) + 1)

    fig = plt.figure(figsize=(9,9))
    
    plt.subplot(2, 1, 1)

    plt.plot(history['loss'], 'red', label='Training loss')
    plt.plot(history['val_loss'], 'green', label='Validation loss')
    plt.legend(loc='upper right')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()

    plt.subplot(2, 1, 2)

    plt.plot(history['acc'], 'orange', label='Training accuracy')
    plt.plot(history['val_acc'], 'blue', label='Validation accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and validation accuracy')

    plt.ylabel('Accuracy score')   
    plt.grid()
    
    plt.tight_layout()

    return fig

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def inverse_normalize(tensor, mean=(0.673, 0.640, 0.604), std=(0.206, 0.206, 0.228)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor

def predict_loader (model, test_loader, device):
    
    lbllist = torch.zeros(0,dtype=torch.long, device = device)
    predlist = torch.zeros(0,dtype=torch.long, device = device)
    problist = torch.zeros(0, dtype = torch.float, device = device)

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
                inputs = data.to(device)
                classes = target.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim = 1)
                probs = F.softmax(outputs, dim = 1)
            
                # Append batch prediction results
                lbllist = torch.cat([lbllist, classes.view(-1)])
                predlist = torch.cat([predlist, preds.view(-1)])
                problist = torch.cat([problist, probs])
    return lbllist, predlist, problist

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

# Added code for Diffferential Privacy Demo:

def dptrain(model, optimizer, loss_fn, train_dl, val_dl, epochs=20, device='cuda', private_training=False, privacy_engine=None, target_delta=3000): #=len(training_loader.dataset.samples)):
    '''
    In Notebook
    Runs training loop for classification problems. Returns Keras-style
    per-epoch history of loss and accuracy over training and validation data.

    Parameters
    ----------
    model : nn.Module
        Neural network model
    optimizer : torch.optim.Optimizer
        Search space optimizer (e.g. Adam)
    loss_fn :
        Loss function (e.g. nn.CrossEntropyLoss())
    train_dl : 
        Iterable dataloader for training data.
    val_dl :
        Iterable dataloader for validation data.
    epochs : int
        Number of epochs to run
    device : string
        Specifies 'cuda' or 'cpu'

    Returns
    -------
    Dictionary
        Similar to Keras' fit(), the output dictionary contains per-epoch
        history of training loss, training accuracy, validation loss, and
        validation accuracy.
    '''

    print('dptrain() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    history['epsilon'] = []

    start_time_sec = time.time()

    for epoch in range(epochs):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss         = 0.0
        num_train_correct  = 0
        num_train_examples = 0

        for batch in train_dl:

            optimizer.zero_grad()

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x.size(0)
            num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]
            
            if private_training:
                epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(delta=target_delta)

        train_acc   = num_train_correct / num_train_examples
        train_loss  = train_loss / len(train_dl.dataset)


        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss       = 0.0
        num_val_correct  = 0
        num_val_examples = 0

        for batch in val_dl:

            x    = batch[0].to(device)
            y    = batch[1].to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x.size(0)
            num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_val_examples += y.shape[0]

        val_acc  = num_val_correct / num_val_examples
        val_loss = val_loss / len(val_dl.dataset)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        epoch_stats = f'Epoch {epoch+1:2d}/{epochs} --- train loss: {train_loss:4.2f}, train acc: {train_acc:4.2f}, val loss: {val_loss:4.2f}, val acc: {val_acc:4.2f}'
        if private_training:
            epoch_stats += f' - epsilon: {epsilon:4.2f}, best alpha: {best_alpha:4.2f}'
            history['epsilon'].append(epsilon)

        print(epoch_stats)

    # END OF TRAINING LOOP


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history

def print_metrics(y_test, y_pred, y_pred_prob, labels):
    accuracy_sc = round(accuracy_score(y_test, y_pred), 3)
    roc_auc_sc = round(roc_auc_score(y_test, y_pred_prob), 3)

    print(classification_report(y_test, y_pred, target_names = labels))

    print('Accuracy score:', accuracy_sc, ' --- ', 'ROC AUC score:', roc_auc_sc)


def predict_loader (model, test_loader, device):
    lbllist = torch.zeros(0,dtype=torch.long, device = device)
    predlist = torch.zeros(0,dtype=torch.long, device = device)
    problist = torch.zeros(0, dtype = torch.float, device = device)

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
                inputs = data.to(device)
                classes = target.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim = 1)
                probs = F.softmax(outputs, dim = 1)
            
                # Append batch prediction results
                lbllist = torch.cat([lbllist, classes.view(-1)])
                predlist = torch.cat([predlist, preds.view(-1)])
                problist = torch.cat([problist, probs])
    return lbllist, predlist, problist


