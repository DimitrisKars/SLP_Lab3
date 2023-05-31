from torch.utils.data import Dataset
from tqdm import tqdm
import re
import numpy as np

class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """
        self.max = 0
        self.data = [self.preprocess(sentence, word2idx) for sentence in tqdm(X, desc="Preprocessing data")]
        self.target= y
        self.word2idx = word2idx
        # self.target


    def preprocess(self, sentence, word2idx):
        """
        Preprocess a single data point - turn a sentence into a sequence
        of word indexes. This could involve tokenizing the sentence, lowercasing,
        perhaps removing punctuation, and finally mapping words to indexes using
        the provided word2idx dictionary.

        Args:
            sentence (str): a sentence string
            word2idx (dict): a dictionary mapping words to indexes

        Returns:
            list: a list of word indexes representing the sentence
        """
        words = re.sub(r'[^\w\s-]|_', '', sentence).lower()
        #words = re.sub(r'[^\w\s]', '', sentence)
        
        words = words.split()

        if (len(words) > self.max):
            self.max = len(words)

        # map words to indexes

        return [word2idx[word] if word in word2idx else word2idx["<unk>"] for word in words]

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """

        # EX3

        example = self.data[index]
        length = len(self.data[index])
        self.data[index] = np.pad(self.target[index], (0, self.max - len(self.data[index])), mode='constant')
        label = self.target[index]


        return example, label, length
        #raise NotImplementedError

