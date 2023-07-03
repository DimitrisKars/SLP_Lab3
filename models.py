import torch
import numpy as np
from torch import nn
# from main import EMB_DIM, n_classes
# from __main__ import vocab_size, EMB_DIM, n_classes
#Double in mean-max pooling than 100
EMB_DIM = 100
n_classes = 3

class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        super(BaselineDNN, self).__init__()

        # EX4
        # 1 - define the embedding layer
        num_embeddings = len(embeddings)
        dim = len(embeddings[0])
        self.embeddings = nn.Embedding(num_embeddings, dim)
        self.output_size = output_size

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        # 3 - define if the embedding layer will be frozen or finetuned
        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(torch.Tensor(embeddings), freeze=True)

        # 4 - define a non-linear transformation of the representations
        # EX5
        self.linear = nn.Linear(dim, 1000)
        self.relu = nn.ReLU()

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        # EX5
        self.output = nn.Linear(1000, output_size)

        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        """
        Our code
        super(BaselineDNN, self).__init__()
        self.output_size = output_size
        self.embeddings = embeddings
        self.trainable_emb = trainable_emb
        #self.hidden_dim = 50

        num_embeddings, embedding_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(num_embeddings,embedding_dim)  # EX4
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))  # Change for mean-only pooling  # EX4
        self.embedding_layer.weight.requires_grad = trainable_emb  # EX4
        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
    #     self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))
    # pretrained_embeddings = torch.tensor(embeddings)
    #     embedding_dim = pretrained_embeddings.size(1)
    #     vocab_size = pretrained_embeddings.size(0)



        # 1 - define the embedding layer
        # EX4
        # self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)


      

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        # Create an embedding layer and initialize weights from pretrained embeddings
        # EX4
        # 3 - define if the embedding layer will be frozen or finetuned
        # self.embedding_layer = nn.Embedding.from_pretrained(pretrained_embeddings)  # EX4
        # self.embedding_layer.requires_grad = trainable_emb
        #self.embedding_layer = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)  # EX4
        self.layers = nn.ModuleList([
            self.embedding_layer,
            nn.Linear(2*embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, n_classes)
        ])

        # 4 - define a non-linear transformation of the representations
        # EX5

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        # EX5
        """

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function defines how the data passes through the network.

        Returns: the logits for each class
        """

        """
         Our code
         # 1 - embed the words, using the embedding layer
         embeddings = self.embedding_layer(x)

        # 2 - construct a sentence representation out of the word embeddings
         representations_mean = torch.sum(embeddings, dim=1)
         for i in range(lengths.shape[0]):
             representations_mean[i] = representations_mean[i] / lengths[i]
         representations_max = torch.max(embeddings, dim=1)[0]
         representations=torch.cat((representations_mean,representations_max),dim=1)


         # 3 - transform the representations to new ones.
         for layer in self.layers[1:]:
             representations = layer(representations)

         # 4 - project the representations to classes using a linear layer
         logits = representations

         return logits
         """

        # 1 - embed the words, using the embedding layer
        # EX6
        embeddings = self.embeddings(x)

        # 2 - construct a sentence representation out of the word embeddings
        # EX6
        # calculate the means
        representations = torch.sum(embeddings, dim=1)
        for i in range(lengths.shape[0]):  # necessary to skip zeros in mean calculation
            representations[i] = representations[i] / lengths[i]

        # 3 - transform the representations to new ones.
        # EX6
        representations = self.relu(self.linear(representations))

        # 4 - project the representations to classes using a linear layer
        # EX6
        logits = self.output(representations)

        return logits



class LSTM(nn.Module):
    def __init__(self, output_size, embeddings, trainable_emb=False, bidirectional=False):

        super(LSTM, self).__init__()
        self.hidden_size = 100
        self.num_layers = 1
        self.bidirectional = bidirectional

        self.representation_size = 2 * \
            self.hidden_size if self.bidirectional else self.hidden_size

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.embeddings = nn.Embedding(num_embeddings, dim)
        self.output_size = output_size

        self.lstm = nn.LSTM(dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)

        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(
                torch.Tensor(embeddings), freeze=True)

        self.linear = nn.Linear(self.representation_size, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x)
        X = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)

        ht, _ = self.lstm(X)

        # ht is batch_size x max(lengths) x hidden_dim
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

      
        # pick the output of the lstm corresponding to the last word
        representations = torch.zeros(batch_size, self.representation_size).float()
        for i in range(lengths.shape[0]):
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            representations[i] = ht[i, last, :]
       
        logits = self.linear(representations)
        
        return logits

