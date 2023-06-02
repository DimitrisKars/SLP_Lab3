import torch

from torch import nn
# from main import EMB_DIM, n_classes
# from __main__ import vocab_size, EMB_DIM, n_classes
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
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()
        self.output_size = output_size
        self.embeddings = embeddings
        self.trainable_emb = trainable_emb
        #self.hidden_dim = 50

        num_embeddings, embedding_dim = embeddings.shape
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)  # EX4
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))  # EX4
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
            #nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, n_classes)
        ])

        # 4 - define a non-linear transformation of the representations
        # EX5

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        # EX5

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function defines how the data passes through the network.

        Returns: the logits for each class
        """

        # 1 - embed the words, using the embedding layer
        embeddings = self.embedding_layer(x)

       # 2 - construct a sentence representation out of the word embeddings
        representation_list=[]
        for i, length in enumerate(lengths):
            non_padded_embeddings = embeddings[i, :length, :]
            representation_example = non_padded_embeddings.mean(dim=0)
            representation_list.append(representation_example)
        
        
        representations=torch.stack(representation_list)
        #print(representations)
        
        # 3 - transform the representations to new ones.
        for layer in self.layers[1:]:
            representations = layer(representations)

        # 4 - project the representations to classes using a linear layer
        logits = representations

        return logits
