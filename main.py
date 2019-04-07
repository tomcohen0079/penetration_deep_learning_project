import os
import pandas as pd
import json
import numpy as np
import glob
from tqdm import tqdm, tqdm_notebook
import torch.utils.data as utils
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import Namespace
import copy
import time
import matplotlib.pyplot as plt

DATA_PATH = '/home/eran_paz/data/'
MASTERKEY_PATH = os.path.join(DATA_PATH, 'masterkey.json')
PENETRATION_PATH = os.path.join(DATA_PATH, 'Penetration estimation.xlsx')
CRASH_DATA = glob.glob(DATA_PATH + '/crash_data' + '/**')

# hyperparameters
args = Namespace(
    seed=1234,
    cuda=False,
    num_epochs=10000,
    learning_rate=1e-3,
    batch_size=128,
    train_p=0.85,
    dropout=0.2,
    threshold_acc=0.25,
    models_dir='Models',
    rounding_factor=-1
)


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


class PenetrationDS(Dataset):
    """create a pytorch dataset object that also   preprocess the data"""

    def __init__(self, crash_path, penetration_path, masterkey_path, transform=None):

        df = self.load_data(crash_path)
        penetration = self.load_predictions(penetration_path, masterkey_path)
        self.data, self.penetration = self.preprocessing(df, penetration)
        self.len, self.timestamp, self.channel = self.data.shape

    def __getitem__(self, index):

        x = self.data[index].reshape(self.channel, self.timestamp).astype(np.float32)
        y = self.penetration[index].astype(np.float32)

        return x, y

    def __len__(self):
        return self.len

    @staticmethod
    def load_data(paths: list):
        df = None
        for path in paths:
            if df is None:
                df = pd.read_csv(path)
            else:
                new_df = pd.read_csv(path)
                df = pd.concat([df, new_df])
        return df.drop(['HEAD_ACX', 'HEAD_ACY', 'HEAD_ACZ'], axis=1)

    @staticmethod
    def load_predictions(path: str, masterkey_path: str):
        # load penetration and replace names that are in dict masterkey
        with open(masterkey_path) as f:
            masterkey = json.load(f)
        penetration = pd.read_excel(PENETRATION_PATH)
        penetration.replace(masterkey, inplace=True)
        # remove rows with strings '-' and '--'
        penetration = penetration.replace(['-', '--'], np.nan).dropna()
        return penetration

    @staticmethod
    def preprocessing(df, penetration):
        # filters the values that are in both  df and penetration and returns them as numpy arrays
        my_x = []
        my_y = []
        for k, group in tqdm_notebook(df.groupby(['CASE_400'])):
            case = group['CASE'].iloc[0]
            x = group.drop(['Time s', 'CASE', 'CASE_400'], axis=1).values  # dropping the timestamp
            y = penetration[penetration['Test No.'] == case]
            if not y.empty:
                my_x.append(x)
                # index zero if more than one case is in the penetration data
                y = y.drop(['Test No.'], axis=1).values[0]
                my_y.append(y.astype(float))
        shape_x = list(my_x[0].shape)
        shape_x[:0] = [len(my_x)]
        shape_y = list(my_y[0].shape)
        shape_y[:0] = [len(my_y)]
        arr_x = np.concatenate(my_x).reshape(shape_x)
        arr_y = np.concatenate(my_y).reshape(shape_y)
        return arr_x, arr_y


# models architecture
# model 1
class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=3,
                out_channels=64,  # Some random number
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

        )
        self.fc1 = nn.Linear(256 * 58, 512)
        self.fc2 = nn.Linear(512, 12)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        #         output = F.leaky_relu(x)
        return out


# model 2
class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First Layer
        # Input
        self.dilated_conv1 = nn.Conv1d(3, 64, kernel_size=3, dilation=1)
        self.relu1 = nn.ReLU()
        self.max1 = nn.MaxPool1d(kernel_size=2)

        # Layer 2
        self.dilated_conv2 = nn.Conv1d(64, 256, kernel_size=3, dilation=2)
        self.relu2 = nn.ReLU()
        self.max2 = nn.MaxPool1d(kernel_size=2)

        # Layer 3
        self.dilated_conv3 = nn.Conv1d(256, 512, kernel_size=3, dilation=4)
        self.relu3 = nn.ReLU()
        self.max3 = nn.MaxPool1d(kernel_size=2)

        # Layer 4
        #         self.dilated_conv2 = nn.Conv1d(512, 512, kernel_size=3, dilation=8)
        #         self.relu2 = nn.ReLU()
        #         self.max2 = nn.MaxPool1d(kernel_size=2)

        self.dropout = nn.Dropout(args.dropout)

        self.fc1 = nn.Linear(512 * 24, 512)
        self.out = nn.Linear(512, 12)

    def forward(self, x):
        """

        :param x: Pytorch Variable, batch_size x
        :return:
        """

        # First layer
        x = self.dilated_conv1(x)
        x = self.relu1(x)
        x = self.max1(x)

        # Layer 2:
        x = self.dilated_conv2(x)
        x = self.relu2(x)
        x = self.max2(x)

        # Layer 3:
        x = self.dilated_conv3(x)
        x = self.relu3(x)
        x = self.max3(x)

        #         print(x.size())
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.out(x)
        return out


# model 3
class RNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=512, num_layers=1, num_features=12):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_features)

    def forward(self, x):
        # resize the input because with lstm you need (batch_size*n_features*timesteps) and our dataloader was created
        # for cnn input
        x = x.view(x.size(0), -1, x.size(1))
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=args.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=args.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class Trainer(object):
    def __init__(self, model, dataset, model_name: str, train_p: float = args.train_p,
                 batch_size: int = args.batch_size,
                 device: str = args.device, num_epochs: int = args.num_epochs,
                 learning_rate: float = args.learning_rate, threshold_acc: float = args.threshold_acc,
                 round_acc=True):

        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.round_acc = round_acc
        self.model_name = model_name
        self.dataloader, self.dataset_sizes = self.create_dataloader(dataset, train_p, batch_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        #         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #             optimizer=self.optimizer, mode='min', factor=0.5, patience=4)
        self.thr_acc = threshold_acc
        self.train_state = {
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'best_loss': -1,
            'best_acc': -1}

    @staticmethod
    def create_dataloader(dataset, train_p, batch_size):

        """
        dataset : pytorch dataset object
        train_p : train percentage
        returns : dataloader dict and dataset sizes dict
        """
        train_size = int(train_p * (len(dataset)))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        datasets = {'train': train_dataset, 'val': val_dataset}
        dataloaders = {x: utils.DataLoader(datasets[x], batch_size=batch_size)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
        return dataloaders, dataset_sizes

    def compute_accuracy_1(self, y_pred, y_target):
        """
        for every prediction vector (1*12), it will round both prediction and target to one decimal AFTER the zero,
        (|y-y_hat|/y_hat) < thr
        we take the average for all the batch

        """
        pred_round = np.round(y_pred.cpu().data, -1)
        targ_round = y_target.cpu().data
        epoch_mean_acc = np.mean(
            np.sum(np.nan_to_num(np.abs(targ_round - pred_round) / pred_round) <= self.thr_acc, axis=1) / y_pred.size(
                1))
        return epoch_mean_acc

    def compute_accuracy_2(self, y_pred, y_target):
        """
        every value that deviates from the target  less than the theresehold will count as an correct prediction,
        (|y-y_hat|/y_hat) < thr
        we take the average for all the batch

        """
        pred_round = y_pred.cpu().data
        targ_round = y_target.cpu().data
        epoch_mean_acc = np.mean(
            np.sum(np.nan_to_num(np.abs(targ_round - pred_round) / pred_round) <= self.thr_acc, axis=1) / y_pred.size(
                1))
        return epoch_mean_acc

    def run_train_loop(self):
        since = time.time()
        print('Training: ' + self.model_name)
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_epoch = None

        for epoch in range(self.num_epochs):

            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:

                if phase == 'train':
                    self.model.train()  # Set model to train mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0
                running_acc = []

                # iterate over data
                for data, target in self.dataloader[phase]:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()

                    # track grad only if train
                    with torch.set_grad_enabled(phase == 'train'):
                        output = self.model(data)
                        loss = self.criterion(output, target)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        running_loss += loss.item() * data.size(0)

                        acc = self.compute_accuracy_1(output, target) if self.round_acc else self.compute_accuracy_2(
                            output, target)
                        running_acc.append(acc)

                # calculate the average loss per epoch and add it to dict
                epoch_loss = running_loss / self.dataset_sizes[phase]
                self.train_state[phase + '_loss'].append(epoch_loss)

                # calculate average accuracy per epoch and add it to dict
                epoch_acc = np.mean(running_acc)
                self.train_state[phase + '_acc'].append(epoch_acc)

                if phase == 'val' and best_epoch is None:
                    best_epoch = epoch_loss

                if phase == 'val' and epoch_loss < best_epoch:
                    best_epoch = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                print('{} Loss: {:.4f} , Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            self.train_state['epoch_index'] += 1
        self.train_state['best_loss'] = best_epoch
        self.train_state['best_acc'] = best_acc
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}, val accuracy: {:4f} '.format(best_epoch, best_acc))

        # load the weights of the model with the lowest eval loss
        self.model.load_state_dict(best_model_wts)

    def plot_performance(self, path: str = args.models_dir):
        # Figure size
        plt.figure(figsize=(15, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(self.train_state["train_loss"], label="train")
        plt.plot(self.train_state["val_loss"], label="val")
        plt.legend(loc='upper right')

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.plot(self.train_state["train_acc"], label="train")
        plt.plot(self.train_state["val_acc"], label="val")
        plt.legend(loc='lower right')

        #         # Save figure
        plt.savefig(os.path.join(path, self.model_name + "_performance.png"))

        # Show plots
        plt.show()

    def save_train_state(self, path: str = args.models_dir):
        """
        path : the path of our model
        """
        # change to include optimizer step
        torch.save(self.model, os.path.join(path, self.model_name + '.pt'))
        torch.save(self.model.state_dict(), os.path.join(path, self.model_name + 'weights.pt'))


def main():
    full_dataset = PenetrationDS(CRASH_DATA, PENETRATION_PATH, MASTERKEY_PATH)

    # make new model directory if it doesnt exists
    if not os.path.isdir(args.models_dir):
        os.mkdir(args.models_dir)
        # set seed
    set_seed(seed=args.seed, cuda=args.cuda)

    # check CUDA
    if torch.cuda.is_available():
        args.cuda = True
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    print('Using CUDA : {}'.format(args.cuda))

    # start training model 1
    trainer_basic = Trainer(model=BasicModel(), dataset=full_dataset, model_name='basic_model', num_epochs=1000)
    trainer_basic.run_train_loop()
    trainer_basic.plot_performance()
    trainer_basic.save_train_state()
    # model 2
    trainer_dilution = Trainer(model=DilatedNet(), dataset=full_dataset, model_name='dilution_model', num_epochs=1000)
    trainer_dilution.run_train_loop()
    trainer_dilution.plot_performance()
    trainer_dilution.save_train_state()
    # model 3

    trainer_lstm = Trainer(model=RNN(), dataset=full_dataset, model_name='lstm_model', num_epochs=10000,
                           learning_rate=0.0001)
    trainer_lstm.run_train_loop()
    trainer_lstm.plot_performance()
    trainer_lstm.save_train_state()


if __name__ == '__main__':
    main()
