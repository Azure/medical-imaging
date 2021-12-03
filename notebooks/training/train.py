import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import mlflow

from azureml.core import Workspace, Dataset, Experiment
from azureml.core.run import Run

SEED = 1

IMG_HEIGHT, IMG_WIDTH = 224, 224
IMG_MEAN = 0.4818
IMG_STD = 0.2357

class Cnn(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 32, kernel_size = 3, stride = 1, padding=  1)
        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64,out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
             
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
                
        self.fc1 = nn.Linear(28*28*128, 256)
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))  # 224 x 224 x 32
        x = F.max_pool2d(x, 2, 2)  # 112 x 112 x 32
        x = F.relu(self.conv2(x))  # 112 x 112 x 64
        x = F.max_pool2d(x, 2, 2)  # 56 x 56 x 64
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))  # 56 x 56 x 128
        x = F.max_pool2d(x, 2, 2)  # 28 x 28 x 128
        x = self.dropout2(x)       
        x = x.view(-1, 28*28*128) # 100.352
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data(workspace, dataset_folder, batch_size):
    """Load the train/val data."""

    # version 1: full dataset, version 2 (latest): reduced train/val to speed up development 
    base_dir = dataset_folder

    # dataset = Dataset.get_by_name(workspace, name='pneumonia', version='latest')
    # dataset.download(target_path=base_dir, overwrite=True)

    torch.manual_seed(SEED)
    transform_train = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = (IMG_MEAN,), std = (IMG_STD,))    
                                ])

    transform_val = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                                    transforms.ToTensor(),   
                                    transforms.Normalize(mean = (IMG_MEAN,), std = (IMG_STD,))    
                                ])

    training_dataset = datasets.ImageFolder(root = os.path.join(base_dir, 'train'), transform = transform_train)
    validation_dataset = datasets.ImageFolder(root = os.path.join(base_dir, 'val'), transform = transform_val)
    test_dataset = datasets.ImageFolder(root = os.path.join(base_dir, 'test'), transform = transform_val)

    training_loader = torch.utils.data.DataLoader(dataset = training_dataset, batch_size = batch_size, shuffle = True, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size = batch_size, shuffle = False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 100, shuffle = False)

    return training_loader, validation_loader


def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=20, device='cuda', run = None):
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


        print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
              (epoch+1, epochs, train_loss, train_acc, val_loss, val_acc))

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        run.log('training loss', train_loss)
        run.log('validation loss', val_loss)
        run.log('training accuracy', train_acc)
        run.log('validation accuracy', val_acc)

        # log primary metric for hyperdrive (using AML SDK)
        
        # check if this is the best epoch so far:
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # save model state related to this epoch
            os.makedirs('./outputs/model', exist_ok=True)
            torch.save(model, os.path.join('./outputs/model', 'pneumonia.pt'))
        
        run.log('best_val_acc', np.float(best_val_acc))
        
    # END OF TRAINING LOOP

    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

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

def main():
    print("Torch version:", torch.__version__)

    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of epochs to train')
    parser.add_argument('--data-folder', type=str, dest='data_folder', default='data', help='data folder mounting point')
    
    # hyperdrive related arguments: 
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='training and validation batch size')
    parser.add_argument('--optimizer', type=str,
                        default='SGD', help='Optimizer: SGD, Adam or RMSprop')
    parser.add_argument('--conv_dropout', type=float,
                        default=0.2, help='dropout parameter for 2nd and 3rd conv-pool layers')
    
    args = parser.parse_args()

    run = Run.get_context()
    workspace = run.experiment.workspace

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    model = Cnn(dropout=args.conv_dropout).to(device)
    training_loader, validation_loader = load_data(workspace, args.data_folder, batch_size=args.batch_size)
    
    mlflow.log_metric('Train imgs', len(training_loader.dataset.samples))
    
    optim_dict = {
        'SGD' : torch.optim.SGD(model.parameters(), lr=args.learning_rate),
        'Adam' : torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        'RMSprop' : torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
    }
    
    optimizer = optim_dict[args.optimizer]

    history = train(model = model,
                device = device,
                optimizer = optimizer,
                loss_fn = nn.CrossEntropyLoss(),
                train_dl = training_loader,
                val_dl = validation_loader,
                epochs = args.epochs,
                run = run)


if __name__ == "__main__":
    with mlflow.start_run():
        main()
