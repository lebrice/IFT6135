import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models import RNN, GRU, FullTransformer 
from models import make_model as TRANSFORMER

# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda") 
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")

def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word


# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."
    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask

print('Loading data from '+ 'data')
raw_data = ptb_raw_data(data_path='data')
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    
    This prevents Pytorch from trying to backpropagate into previous input 
    sequences when we use the final hidden states from one mini-batch as the 
    initial hidden states for the next mini-batch.
    
    Using the final hidden states in this way makes sense when the elements of 
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)

def run_epoch(model, data, is_train=False, lr=1.0):
    """
    One epoch of validation
    """
    model.eval()
    epoch_size = ((len(data) // model.batch_size) - 1) // model.seq_len
    start_time = time.time()
    if not isinstance(model, FullTransformer):
        hidden = model.init_hidden()
        hidden = hidden.to(device)
    costs = 0.0
    iters = 0
    losses = torch.zeros(model.seq_len, device=device)
    num_batches = 0
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # LOOP THROUGH MINIBATCHES
    for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
        print(step)
        if isinstance(model, FullTransformer):
            batch = Batch(torch.from_numpy(x).long().to(device))
            model.zero_grad()
            outputs = model.forward(batch.data, batch.mask).transpose(1,0)
            #print ("outputs.shape", outputs.shape)
        else:
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            outputs, hidden = model(inputs, hidden)

        targets = torch.from_numpy(y.astype(np.int64)).contiguous().to(device)
        tt = torch.squeeze(targets)

        # LOSS COMPUTATION
        # This line currently averages across all the sequences in a mini-batch 
        # and all time-steps of the sequences.
        # For problem 5.3, you will (instead) need to compute the average loss 
        # at each time-step separately. 
        loss = loss_fn(outputs.contiguous().permute(1, 2, 0), tt)
        # costs += loss.data.item() * model.seq_len
        losses += loss.detach().mean(0)
        num_batches += 1
        iters += model.seq_len
    return np.exp(costs / iters), losses / num_batches

def get_best_model(model_type: str) -> nn.Module:
    model: nn.Module = None
    if model_type == 'RNN':
        model = RNN(emb_size=200, hidden_size=1500, 
                        seq_len=35, batch_size=20,
                        vocab_size=vocab_size, num_layers=2, 
                        dp_keep_prob=0.35)
        model.load_state_dict(torch.load('./4_1_a/best_params.pt', map_location=device))
    elif model_type == 'GRU':
        model = GRU(emb_size=200, hidden_size=1500, 
                seq_len=35, batch_size=20,
                vocab_size=vocab_size, num_layers=2, 
                dp_keep_prob=0.35)
        model.load_state_dict(torch.load('./4_1_b/best_params.pt', map_location=device))
    elif model_type == 'TRANSFORMER':
        model = TRANSFORMER(vocab_size=vocab_size, n_units=512, 
                            n_blocks=6, dropout=1.-0.9)
        model.batch_size=128
        model.seq_len=35
        model.vocab_size=vocab_size
        model.load_state_dict(torch.load('./4_1_c/best_params.pt'))
    return model

def main():        
    for model_type in ['RNN', 'GRU', 'TRANSFORMER']:
        
        model = get_best_model(model_type)
        model = model.to(device)

        _, losses = run_epoch(model, valid_data)

        losses = losses.detach().cpu().numpy()
        plt.clf()
        plt.plot(range(len(losses)), losses)
        plt.title('Average loss at each time-step within validation sequences')
        plt.xlabel('Time-step')
        plt.ylabel('Average loss')
        plt.savefig(f'Q5_1_loss_at_time_steps_{model_type}.jpg')

if __name__ == "__main__":
    main()