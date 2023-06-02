import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader,TensorDataset

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
import torch.optim as optim
from torch import nn

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.twitter.27B.100d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 100

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "Semeval2017A"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
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
n_classes = len(le.classes_) # EX1 - LabelEncoder.classes_.size

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

for i in range(5):
    print(train_set[i])
    
# X_train = torch.tensor(X_train)
# y_train = torch.tensor(y_train)
# X_test = torch.tensor(X_test)
# y_test = torch.tensor(y_test)

# EX7 - Define our PyTorch-based DataLoader
# dataset_train = TensorDataset(X_train, y_train)
# dataset_test = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)  # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)  # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
model = BaselineDNN(output_size=n_classes,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
criterion = nn.CrossEntropyLoss()  # EX8

parameters = []  # EX8
for p in model.parameters():
    if p.requires_grad:
        parameters.append(p)
        
optimizer = optim.Adam(parameters, lr=0.0001)  # EX8

#############################################################################
# Training Pipeline
#############################################################################
total_train_loss = []
total_test_loss = []
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_pred, y_train_gold) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)

    total_train_loss.append(train_loss)

    test_loss, (y_test_pred, y_test_gold) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)

    total_test_loss.append(test_loss)

accuracy_train = 0
f1_score_train = 0
recall_score_train = 0
for true, pred in zip(y_train_gold, y_train_pred):
    accuracy_train += accuracy_score(true, pred)
    f1_score_train += f1_score(true, pred, average="macro")
    recall_score_train += recall_score(true, pred, average="macro")

accuracy_train /= len(y_train_gold)
f1_score_train /= len(y_train_gold)
recall_score_train /= len(y_train_gold)

print("Accuracy score for train set:", accuracy_train)
print("F1-score for train set:", f1_score_train)
print("Recall score for train set:", recall_score_train)

accuracy_test = 0
f1_score_test = 0
recall_score_test = 0
for true, pred in zip(y_test_gold, y_test_pred):
    accuracy_test += accuracy_score(true, pred)
    f1_score_test += f1_score(true, pred, average="macro")
    recall_score_test += recall_score(true, pred, average="macro")

accuracy_test /= len(y_test_gold)
f1_score_test /= len(y_test_gold)
recall_score_test /= len(y_test_gold)

print("Accuracy score for test set:", accuracy_test)
print("F1-score for test set:", f1_score_test)
print("Recall score for test set:", recall_score_test)


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
