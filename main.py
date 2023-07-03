import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN, LSTM
from attention import SimpleSelfAttentionModel, MultiHeadAttentionModel, TransformerEncoderModel
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
import torch.optim as optim
from torch import nn

import matplotlib.pyplot as plt


from training import torch_train_val_split, get_metrics_report
from early_stopper import EarlyStopper

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.twitter.27B.100d.txt")

# 2 - set the correct dimensionality of the embeddings
# Double in mean-max than 100
EMB_DIM = 100

EMB_TRAINABLE = False
BATCH_SIZE = 128
# EPOCHS = 3
DATASET = "Semeval2017A"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible GPU, use it; otherwise, use the CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)
vocab_size = len(word2idx.keys())

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)  # EX1
y_test = le.transform(y_test)  # EX1
n_classes = len(le.classes_)  # EX1 - LabelEncoder.classes_.size

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

for i in range(5):
    print(train_set[i])

# EX7 - Define our PyTorch-based DataLoader
train_loader, val_loader = torch_train_val_split(train_set, BATCH_SIZE, BATCH_SIZE)
# train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)  # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)  # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################

models = {}
# model_epochs = {}
# models["Baseline"] = BaselineDNN(output_size=n_classes,  # EX8
#                                  embeddings=embeddings,
#                                  trainable_emb=EMB_TRAINABLE)
#
#
# models["LSTM"] = LSTM(output_size=n_classes,
#                       embeddings=embeddings,
#                       trainable_emb=EMB_TRAINABLE)
#
# models["LSTM_Bidirectional"] = LSTM(output_size=n_classes,
#                                     embeddings=embeddings,
#                                     trainable_emb=EMB_TRAINABLE, bidirectional=True)
#
# models["SimpleSelfAttention"] = SimpleSelfAttentionModel(output_size=n_classes,
#                                                          embeddings=embeddings)

# models["MultiHead"] = MultiHeadAttentionModel(output_size=n_classes,
#                                               embeddings=embeddings)

models["TransformerEncoder"] = TransformerEncoderModel(output_size=n_classes, embeddings=embeddings)

model_epochs = {
    "Baseline": 50,
    "LSTM": 20,
    "LSTM_Bidirectional": 20,
    "SimpleSelfAttention": 20,
    "MultiHead": 10,
    "TransformerEncoder": 10,
}

for m in models.keys():
    model = models[m]
    EPOCHS = model_epochs[m]
    # Move the model to GPU
    model.to(DEVICE)
    print(model)

    # We optimize ONLY those parameters that are trainable (p.requires_grad==True)
    criterion = nn.CrossEntropyLoss()  # EX8

    parameters = []  # EX8
    for p in model.parameters():
        if p.requires_grad:
            parameters.append(p)

    # Move the parameters to GPU
    for i in range(len(parameters)):
        parameters[i] = parameters[i].to(DEVICE)

    optimizer = optim.Adam(parameters, lr=0.0001)  # EX8

    #############################################################################
    # Training Pipeline
    #############################################################################
    save_path = f'{DATASET}_{model.__class__.__name__}.pth'
    early_stopper = EarlyStopper(model, save_path, patience=5)


    total_train_loss = []
    total_test_loss = []
    total_valid_loss = []
    print("Scores for ", m, " model")
    for epoch in range(1, EPOCHS + 1):
        # train the model for one epoch
        train_dataset(epoch, train_loader, model, criterion, optimizer)

        # evaluate the performance of the model, on both data sets
        train_loss, (y_train_pred, y_train_gold) = eval_dataset(train_loader,
                                                                model,
                                                                criterion)

        total_train_loss.append(train_loss)

        valid_loss, (y_valid_pred, y_valid_gold) = eval_dataset(val_loader,
                                                                model,
                                                                criterion)

        # total_test_loss.append(test_loss)
        total_valid_loss.append(valid_loss)

        print(f"\n===== EPOCH {epoch} ========")
        print(f'\nTraining set\n{get_metrics_report(y_train_gold, y_train_pred)}')
        print(f'\nValidation set\n{get_metrics_report(y_valid_gold, y_valid_pred)}')

        if early_stopper.early_stop(valid_loss):
            print('Early Stopping was activated.')
            print(f'Epoch {epoch}/{EPOCHS}, Loss at training set: {train_loss}\n\tLoss at validation set: {valid_loss}')
            print('Training has been completed.\n')
            break

    test_loss, (y_test_pred, y_test_gold) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)
    print(f'\nTest set\n{get_metrics_report(y_test_gold, y_test_pred)}')

    plt.figure()
    plt.xlabel("Epoch")
    plt.title("Train loss")
    plt.plot(total_train_loss)
    plt.savefig("train_loss.svg", format='svg')

    plt.figure()
    plt.xlabel("Epoch")
    plt.title("Test loss")
    plt.plot(total_test_loss)
    plt.savefig("test_loss.svg", format='svg')
