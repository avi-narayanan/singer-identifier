# custom callback to plot and monitor training
# accuracy_key -> what shows up at the end of the model.fit progress bar

from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

class ModelLogger(Callback):
    def __init__(self, accuracy_key):
        self.i = 0
        self.x = []
        self.data = {
            'losses':[],
            'val_losses':[],
            'acc':[],
            'val_acc':[]
        }
        self.fig = plt.figure()
        self.batch_log = {
            'losses':[],
            'acc':[]
        } 
        self.log_x = []
        self.log_count = 0
        self.acc_key = accuracy_key
    
    def copy(self, other):
        self.data = other.data
        self.x = other.x
        self.i = other.i
        self.debug = None
        self.batch_log = other.batch_log
        self.log_x = other.log_x
        self.log_count = other.log_count

        
    def windowed_mean(self, x, w):
        x = np.array(x[:len(x) - (len(x) % w) ])
        return np.mean(x.reshape(-1, w), axis=1)
    
    def plot(self):
        f, ax = plt.subplots(2, 2)
        f.set_size_inches(20,12)

        ax[0][0].plot(self.batch_log['losses'], label="loss", color='red')
        ax[0][0].set_title("Batch Loss")
        ax[0][0].legend()
        
        ax[0][1].plot(self.batch_log['acc'], label="accuracy", color='orange')
        ax[0][1].legend()
        ax[0][1].set_title("Batch Accuracy")
        
        ax[1][0].set_yscale('log')
        ax[1][0].plot(self.data['losses'], label="loss", marker='.')
        ax[1][0].plot(self.data['val_losses'], label="val_loss", marker='.')
        ax[1][0].legend()
        ax[1][0].set_title("Epoch Loss")
        
        ax[1][1].plot(self.data['acc'], label="accuracy", marker='.')
        ax[1][1].plot(self.data['val_acc'], label="validation accuracy", marker='.')
        ax[1][1].legend()
        ax[1][1].set_title("Epoch Accuracy")
        
        plt.show();
    
    def on_epoch_end(self, epoch, logs={}):
        self.x.append(self.i)
        self.data['losses'].append(logs.get('loss'))
        self.data['val_losses'].append(logs.get('val_loss'))
        self.data['acc'].append(logs.get(self.acc_key))
        self.data['val_acc'].append(logs.get('val_' + self.acc_key))
        self.i += 1
        clear_output(wait=True)
        self.debug = logs
        self.plot()
        self.reset_batch_log()
    
    def on_batch_end(self, batch, logs={}):
        current = logs.get('loss')
        self.batch_log['losses'].append(current)
        self.batch_log['acc'].append(logs.get(self.acc_key))
        self.log_count += 1
        self.log_x.append(self.log_count)
        if current == None:
            self.reset_batch_log()  
    
    def reset_batch_log(self):
        self.batch_log['losses'] = []
        self.batch_log['acc'] = []
        self.log_x = []
        self.log_count = 0
          