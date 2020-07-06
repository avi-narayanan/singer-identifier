import numpy as np
import os
import glob
import datetime
from random import shuffle
from tensorflow.keras.utils import Sequence
from math import ceil, floor

class FingerprintingDataGenerator(Sequence):
    """
    Generator will find all files with the given file extension and return a dataset (X,y) where
            X is pair of data points from the dataset
            y is 1 if the pair belongs to the same class and 0 if different
    Data is expected to be organized into folders where each folder is one class as shown below 
    |- class_label_1
    |    |- example1.npy
    |    |- example2.npy
    |    .
    |    .
    |- class_label_2
    |    |- example1.npy
    |    |- example2.npy
    .    .
    .    .
    """
    
    def __init__(self, data_dir, batch_size, n_days=None, sampling_mode='all', num_of_steps_per_epoch = None, 
                 file_extension='.npy', exclude_class=None, shuffle=True,verbose=False, n_threads=None):
        """ 
        Parameters:
        ----------------
        data_dir : each folder in data_dir is a class 
        batch_size : number of data samples per batch
        n_days : only consider files that have been modified in the last n_days
        sampling_mode : allowed values, 'all' and 'balanced'
            'all' will return every combination of data points possible
            'balanced' will randomly sample pairs such that every batch has equal number of y=0 and y=1
        num_steps_per_epoch : must be provided if sampling mode is 'balanced'
        file_extension : file type of each example in the folder, by default '.npy'
        exclude_class : class label whose data should not be included in the generator
            If the class label provided does not exist, it is ignored
        
        """
        self.n_threads = n_threads
        self.verbose = verbose
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.file_extension = file_extension
        self.sampling_mode = sampling_mode
        if self.sampling_mode not in ['all','balanced']:
            raise ValueError("Sampling mode must be either 'all' or 'balanced'")
        # Get all the file paths in the data_dir with the extension provided 
        self.files = np.array(glob.glob(os.path.join(data_dir,'*/*'+file_extension)))
        today = datetime.datetime.now()
        if n_days is not None:
            self.files = list(filter(lambda x: self.num_days(today, x)<=n_days, self.files))
        if exclude_class is not None:
            self.files = list(filter(lambda x: os.path.basename(os.path.dirname(x)) != exclude_class, self.files))
        self.num_files = len(self.files)
        
        if self.sampling_mode=='all':
            """
            If there are n files, there are Σn possible unique pairs of files
            We go through all the files by filling out a hypothetical matrix N where 
               N(i,j) = multiplier[i]*n_cols + offset[j]
            The dimension of N : 
               num_files x (num_files+1)/2 if num_files is odd
               num_files/2 x (num_files + 1) if num_files is even
            """
            self.__n_rows = self.num_files if self.num_files%2!=0 else self.num_files//2
            self.__n_cols = self.num_files+1 if self.num_files%2==0 else (self.num_files+1)//2
            self.__multiplier = np.arange(self.__n_rows)
            self.__offsets = np.arange(self.__n_cols)
            self.num_of_steps_per_epoch = ceil(len(self.files) / self.batch_size)
            if shuffle:
                np.random.shuffle(self.__offsets)
                np.random.shuffle(self.__multiplier)
        if self.sampling_mode == 'balanced':
            if num_of_steps_per_epoch is None:
                raise ValueError("Number of steps per epoch must be provided when sampling mode is 'balanced'")
            else:
                self.num_of_steps_per_epoch = num_of_steps_per_epoch
                self.classes = {}
                for f in self.files:
                    class_name = os.path.basename(os.path.dirname(f))
                    if class_name not in self.classes:
                        self.classes[class_name] = [f]
                    else:
                        self.classes[class_name].append(f)
       
    def __len__(self):
        """Returns the number of steps per epoch"""
        return self.num_of_steps_per_epoch
        
    def __getitem__(self, index):
        if self.sampling_mode == 'all':
            comb_number_list = self.get_combination_numbers(index*self.batch_size)
            file_indices = [self.get_file_indices(n) for n in comb_number_list]
            X,Y = self.get_files(file_indices)
        elif self.sampling_mode == 'balanced':
            X,Y = self.get_balanced_batch()
        return X,Y
    
    def get_balanced_batch(self):
        """
        Generate a batch that has equal number of positive and negative samples
        """
        Y = np.zeros(shape = (self.batch_size))
        Y[np.random.choice(range(self.batch_size),self.batch_size//2, replace=False)] = 1
        class_labels = list(self.classes.keys())
        X1 = []
        X2 = []
        for i in range(self.batch_size):
            f1,f2='',''
            if Y[i] == 1:
                key = np.random.choice(class_labels)
                f1,f2 = np.random.choice(self.classes[key],2)
            elif Y[i] == 0:
                k1,k2 = np.random.choice(class_labels,2, replace=False)
                f1 = np.random.choice(self.classes[k1])
                f2 = np.random.choice(self.classes[k2])
            try:
                x_i = np.load(f1)
                x_i = np.reshape(x_i, x_i.shape + tuple([1])) # channels last
                X1.append(x_i)
                x_j = np.load(f2)
                x_j = np.reshape(x_j, x_j.shape + tuple([1])) # channels last
                X2.append(x_j)
            except FileNotFoundError:
                if self.verbose:
                    print("FileNotFoundError: Skipping "+file)
                continue
        del x_i,x_j,f1,f2
        X1 = np.array(X1)
        X2 = np.array(X2)
        Y = np.array(Y)
        return [X1,X2],Y
    
    def load_file(self, file_index):
        """
        Returns a single numpy array for the given file index where file path is files[file_index]
        """
        try:
            f = self.files[file_index]
            x = np.load(f)
            x = np.reshape(x, x.shape + tuple([1])) # channels last
            return x
        except FileNotFoundError:
            if self.verbose:
                print("FileNotFound Error: Skipping "+f)
    
    def get_files(self, file_indices):
        """
        Return a dataset of numpy arrays X,Y 
            X is pair of data points from the dataset
            y is 1 if the pair belongs to the same class and 0 if different
        Parameters
        ------------
        file_indices : a list of tuples (i,j) where files[i] and files[j] are the
            pair of data points to be read
        n_threads : number of threads to use, if not specified files are loaded sequentially
        
        """
        X1 = []
        X2 = []
        Y = []
        if self.n_threads is not None:
            from multiprocessing import Pool
            with Pool(processes=self.n_threads) as pool:
                X1 = pool.map(self.load_file,[i for i,j in file_indices])
            with Pool(processes=self.n_threads) as pool:
                X2 = pool.map(self.load_file,[j for i,j in file_indices])
            Y = [1 if os.path.basename(os.path.dirname(self.files[i]))==os.path.basename(os.path.dirname(self.files[j])) else 0 for i,j in file_indices]
        else:           
            for i,j in file_indices:
                i_label = os.path.basename(os.path.dirname(self.files[i])) # ./Spectrograms/train/Avinash/record1.npy returns Avinash
                j_label = os.path.basename(os.path.dirname(self.files[j]))
                try:
                    x_i = np.load(self.files[i])
                    x_i = np.reshape(x_i, x_i.shape + tuple([1])) # channels last
                    X1.append(x_i)
                    x_j = np.load(self.files[j])
                    x_j = np.reshape(x_j, x_j.shape + tuple([1])) # channels last
                    X2.append(x_j)
                    Y.append(1 if i_label==j_label else 0)
                except FileNotFoundError:
                    if self.verbose:
                        print("FileNotFoundError: Skipping "+file)
                    continue
            del x_i,x_j,i_label,j_label
        X1 = np.array(X1)
        X2 = np.array(X2)
        Y = np.array(Y)
        return [X1,X2],Y
    
    def get_file_indices(self, n):
        """
        Returns the indexes of the individual files that generated the combination number 'n'
        Can be reworded as, find i and j such that Σi +j = n and i>=j
        Take the matrix M:
          0 1 2 .
          -------
        0|0
        1|1 2
        2|3 4 5
        .|. . . .
        Function returns row index (i) and column index (j) such that M(i,j) = n
        """
        
        i = 0
        j = 0
        possible_i = floor(np.sqrt(2*n))
        if possible_i**2 + possible_i <= (2*n):
            i = possible_i
        else:
            i = possible_i - 1
        j = n - (i*(i+1))//2
        return (i,j)
    
    def get_combination_numbers(self,index):
        """
        Returns a list of 'batch_size' number of combination numbers which are used as input to get_file_indices
        Example - 
            if index=6 and batch_size=3, the function returns the 6th, 7th, and 8th items (zero indexed) from 
            matrix N (see above) assuming N is populated from left to right, top to bottom
        """
        
        i = (index-1)//self.__n_cols
        j = (index-1)%self.__n_cols
        counter = 0
        output = []
        while counter<self.batch_size:
            if j==self.__n_cols:
                j = 0
                i = i + 1
            if i==self.__n_rows:
                break
            output.append(self.__multiplier[i]*self.__n_cols + self.__offsets[j])
            j += 1
            counter += 1
        return output
    
    def num_days(self, today, file_path):
        """
        Return the number of days passed since file_path was last modified
        Parameters
        ---------------
        today : datetime object of today's date
        file_path : file whose last modified date is under consideration
        
        """
        mod_date = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        diff = today - mod_date
        return diff.days
    
    def pseudo_random(self, n, multiplier):
        """Yield all numbers in the range(0,n) in a somewhat random order"""
        
        multiplier = np.uint16(multiplier)
        offsets = np.arange(multiplier, dtype=np.uint16)
        np.random.shuffle(offsets)
        numbers = []
        for offset in offsets:
            i = 0
            while (i*multiplier + offset)<n:
                yield(i*multiplier + offset)
                i = i+1
        
class ClassificationDataGenerator(Sequence):
    """
    Generator will find all files with the given file extension and return a dataset (X,y) where
            X is a data point from the dataset
            y is (0,1) if mode is 'binary' and one-hot encoded vector is mode is 'multi-class'
    Data is expected to be organized into folders where each folder is one class as shown below 
    |- class_label_1
    |    |- example1.npy
    |    |- example2.npy
    |    .
    |    .
    |- class_label_2
    |    |- example1.npy
    |    |- example2.npy
    .    .
    .    .
    """
    
    
    def __init__(self, data_dir, batch_size, n_files = None, n_days = None, file_extension = '.npy', 
                 mode = 'binary', positive_class = None, labels = None, exclude_class=None, shuffle = True, verbose=False):
        """
        Parameters
        ---------------
        data_dir : each folder in the data_dir is a class
        batch_size : number of data samples per batch
        n_files : total number of files to return, must be lesser than total number of files in data_dir
            If shuffle = False, returns the first n files, else return n random files in the data_dir
        n_days : only consider files that have been modified in the last n_days, n_files filter is applied first
        file_extension : file type of each example in the folder, by default '.npy'
        mode : can be 'binary' or 'multi-class'
        positive_class : if mode is binary, this decides which is positive class (y=1) and 
                        all other classes are grouped into negative class (y=0)
        labels : it is the list of classes to consider for the multi class classifier. Must be provided 
                if mode is multi-class
        
        """
        self.verbose = verbose
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.file_extension = file_extension
        self.mode = mode
        self.positive_class = positive_class
        self.labels = labels
        if self.mode not in set(['binary','multi-class']):
            raise ValueError("Mode must be either binary or multi-class")
        if self.mode == 'binary' and positive_class is None:
            raise ValueError("Positive class label must be provided when mode is binary")
        if self.mode == 'multi-class' and self.labels is None:
            raise ValueError("Labels must be provided when mode is multi-class")
        # Get all the file paths in the data_dir with the extension provided 
        self.files = np.array(glob.glob(os.path.join(data_dir,'*/*'+file_extension)))
        if shuffle:
            np.random.shuffle(self.files)
        if n_files is not None:
            if n_files>len(self.files):
                raise ValueError("Number of files provided is greater than the number of files in directory. Use value less than" 
                             + str(len(self.files)))
            self.files = self.files[:n_files]
        today = datetime.datetime.now()
        if n_days is not None:
            self.files = list(filter(lambda x: self.num_days(today, x)<=n_days, self.files))
        if exclude_class is not None:
            self.files = list(filter(lambda x: os.path.basename(os.path.dirname(x)) != exclude_class, self.files))
        if batch_size>len(self.files):
            raise ValueError("Batch size can not be larger than total number of files")
                  
    def __len__(self):
        """Returns the number of batches per epoch"""
        return ceil(len(self.files) / self.batch_size)
    
    def __getitem__(self, index):
        file_list = self.files[index*self.batch_size:(index+1)*self.batch_size]   
        X,Y = self.get_data(file_list)
        return X,Y
    
    def get_data(self, file_list):
        X = []
        Y = []
        for file in file_list:
            label = os.path.basename(os.path.dirname(file)) # ./Spectrograms/train/Avinash/record1.npy returns Avinash
            try:
                t = np.load(file)
                X.append(np.reshape(t, (t.shape[0],t.shape[1],1))) # channels last
                Y.append(self.get_y(label))
            except FileNotFoundError:
                if self.verbose:
                    print("FileNotFoundError: Skipping "+file)
                continue
            except ValueError:
                if self.verbose:
                    print("Example is not in labels provided: Skipping "+file)
        X = np.array(X)
        Y = np.array(Y)
        return X,Y
    
    def get_y(self, label):
        out = None
        if self.mode=='binary':
            out = 1 if label==self.positive_class else 0
        elif self.mode=='multi-class':
            out = np.zeros(shape=(len(self.labels)))
            out[self.labels.index(label)] = 1
        return out
    
    def num_days(self, today, file_path):
        """
        Return the number of days passed since file_path was last modified
        Parameters
        ---------------
        today : datetime object of today's date
        file_path : file whose last modified date is under consideration
        
        """
        mod_date = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        diff = today - mod_date
        return diff.days