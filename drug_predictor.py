from ast import arg
import pandas as pd
import numpy as np
import itertools
import torch
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import argparse


def count_kmers(read, k):
    """Count kmer occurrences in a given read.

    Parameters
    ----------
    read : string
        A single DNA sequence.
    k : int
        The value of k for which to count kmers.

    Returns
    -------
    counts : dictionary, {'string': int}
        A dictionary of counts keyed by their individual kmers (strings
        of length k).

    Examples
    --------
    >>> count_kmers("GATGAT", 3)
    {'ATG': 1, 'GAT': 2, 'TGA': 1}
    """
    # Start with an empty dictionary
    counts = {}
    # Calculate how many kmers of length k there are
    num_kmers = len(read) - k + 1
    # Loop over the kmer start positions
    for i in range(num_kmers):
        # Slice the string to get the kmer
        kmer = read[i:i+k]
        # Add the kmer to the dictionary if it's not there
        if kmer not in counts:
            counts[kmer] = 0
        # Increment the count for this kmer
        counts[kmer] += 1
    # Return the final counts
    return counts

def read_markers(marker_seq_file):
    """
    read fastafile return list of sequences
    """
    marker_seq = []
    names = []
    i = -1
    with open(marker_seq_file, 'r') as marker:
        for line in marker:
            if line.startswith(">"):
                names.append(line[:-1])
                i += 1
                marker_seq.append("")
                marker_seq[i] = ''
            else:
                marker_seq[i] += line.strip()
        marker.close()
    return names, marker_seq

# create embeddings
def kmer_freq_embed(seq_list, k=4):

    def generate_all_kmer(k=4):
        """
        input: k-mer length
        output: all k-mers list
        """
        a = [['A', 'T', 'C', 'G']]*k
#     print(a)
        return ["".join(list(x)) for x in list(itertools.product(*a))]

    vocab = generate_all_kmer()
    embeds = []
    for seq in seq_list:
        embed = []
        kmers = count_kmers(seq, k)
        for v in vocab:
            if kmers.get(v):
                embed.append(kmers.get(v))
            else:
                embed.append(0)
        embeds.append(embed)
    return np.array(embeds).reshape(len(seq_list),1,16,16)
    
# define model
# feature_map -> cnn_layer -> linear_proj -> u/mg
class cnn_net(torch.nn.Module):   
    def __init__(self):
        super(cnn_net, self).__init__()

        self.cnn_layers = torch.nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=0)
        self.pooling = torch.nn.MaxPool2d(2,2)
        self.layer1 = torch.nn.Linear(36*3, 36*3)
        self.linear_layers = torch.nn.Linear(36*3, 1)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.ReLU()(self.layer1(x))
        x = self.linear_layers(x)
        return x


# train with validation
def train(train_data, train_targets, test_data, test_targets, epoch_num=30):
    model = cnn_net()
    def val():
        with torch.no_grad():
            pred = model(torch.Tensor(test_data))
            loss = criterion(pred, torch.Tensor(test_targets))
        return loss
    # train
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    # epoch_num = 30
    history = {'train':[],'test':[]}
    for epoch in range(epoch_num):
        optimizer.zero_grad()
        outputs = model(torch.Tensor(train_data))
        loss = criterion(outputs, torch.Tensor(train_targets))
        loss.backward()
        optimizer.step()
        training_loss = loss.item()
        val_loss = val().item()
        history['train'].append(training_loss)
        history['test'].append(val_loss)
        # if (epoch+1)%50 == 0:
        print(f'[Epoch {epoch + 1}:] training loss: {training_loss:.3f} test loss: {val_loss:.3f}')
    print('Finished Training')
    sns.lineplot(data=history)
    plt.savefig("train_log.png",dpi=300)
    plt.clf()
    return model


    # check training log
    ax = sns.lineplot(data=history)
    plt.savefig("train_log.png",dpi=300)

def inference(model, input, taregt=None, exp_name=None):
    if exp_name:
        with torch.no_grad():
            pred = model(torch.Tensor(input)).numpy().reshape(-1,)
        print(f"{exp_name} results:")
        print(f"Spearman's rho: {stats.spearmanr(taregt.reshape(-1,), pred)[0]}")
        print(f"Pearsons'r: {stats.pearsonr(taregt.reshape(-1,), pred)[0]}")
        ax = sns.scatterplot(x=taregt.reshape(-1,), y=pred, label=exp_name)
        ax.set_xlabel("targets")
        ax.set_ylabel("prediction")
        ax.set_title(f"{exp_name}")
        plt.savefig(f"{exp_name}.png",dpi=300)
        plt.clf()
        return 0
    else:
        with torch.no_grad():
            pred = model(torch.Tensor(input)).numpy().reshape(-1,)
        return pred

def getparser():
    parser = argparse.ArgumentParser(
        description="""training or run prediction of drug resistance using marker sequences, score is log(concentration)"""
    )
    parser.add_argument(
        "seq",
        help="marker sequences in fasta file",
        type=str,
    )
    parser.add_argument(
        "mode",
        help="train or inference",
        type=str,
        choices=["train", "inference"]
    )
    parser.add_argument("--drug_data", help="experiment tested drug resistance value", type=str)
    parser.add_argument("--model_name", help="experiment name, for the of checkpoint to save or load", type=str)
    parser.add_argument("--epoch_num", help="train the model for how many epoches, default is 30.", default=30)
    return parser.parse_args()


if __name__ == '__main__':
    args = getparser()
    seq_file = args.seq
    mode = args.mode
    drug_data = args.drug_data
    model_name = args.model_name
    epoch_num = args.epoch_num

    # names, d1d2_seq = read_markers("D1D2_107.fas")
    names, seq_list = read_markers(seq_file)
    input_data = kmer_freq_embed(seq_list)


    if mode == 'train':
        # drug = pd.read_csv("drug_data.txt",sep=' ',index_col='FIELD_LABELS')
        drug = pd.read_csv(drug_data, sep=' ',index_col='FIELD_LABELS')
        x = pd.DataFrame(np.concatenate((np.array(drug.index).reshape(-1,1),np.log(drug.values)),axis=1)) # make array-like
        targets = x[1].values.astype(float)

        train_num = int(len(targets)*0.8)
        train_data, train_targets = input_data[:train_num,:,:], targets[:train_num].reshape(-1,1)
        test_data, test_targets = input_data[train_num:,:,:], targets[train_num:].reshape(-1,1)
        model = train(train_data, train_targets, test_data, test_targets)
        inference(model, train_data, train_targets, exp_name='train')
        inference(model, test_data, test_targets, exp_name='test')
        torch.save(model.state_dict(), f"{model_name}.pth")
        print(f"model saved to {model_name}.pth!")

    if mode == 'inference':
        model = cnn_net()
        model.load_state_dict(torch.load(f"{model_name}.pth"))
        model.eval()
        pred = inference(model, input_data).tolist()
        of = open(f"{seq_file}.dscore", "w+")
        for name, value in zip(names, pred):
            of.write(f"{name}\t{str(round(value, 5))}\n")
        of.close()
        print(f"Finished prediction using {model_name}.pth!")