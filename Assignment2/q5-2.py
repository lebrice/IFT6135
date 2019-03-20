import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from models import FullTransformer
from q5 import get_best_model, train_data, ptb_iterator, repackage_hidden, Batch, device

def run_batch(model, data):
    model.eval()
    if not isinstance(model, FullTransformer):
        hidden = model.init_hidden()
        hidden = hidden.to(device)
    
    loss_fn = nn.CrossEntropyLoss()

    # LOOP THROUGH MINIBATCHES
    x, y = next(ptb_iterator(data, model.batch_size, model.seq_len))

    model.zero_grad()

    if isinstance(model, FullTransformer):
        batch = Batch(torch.from_numpy(x).long().to(device))
        outputs = model.forward(batch.data, batch.mask).transpose(1,0)
        #print ("outputs.shape", outputs.shape)
    else:
        inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)
        # hidden = repackage_hidden(hidden)
        outputs, new_hidden = model(inputs, hidden)

    targets = torch.from_numpy(y.astype(np.int64)).contiguous().to(device)
    tt = torch.squeeze(targets.view(-1, model.batch_size * model.seq_len))

    def register_grad_hook(tensor):
        def hook(grad):
            tensor.hidden_grad = grad
        tensor.register_hook(hook)

    for hidden_state in model.hidden_states:
        register_grad_hook(hidden_state)

    # LOSS COMPUTATION
    # This line currently averages across all the sequences in a mini-batch 
    # and all time-steps of the sequences.
    # For problem 5.3, you will (instead) need to compute the average loss 
    # at each time-step separately. 
    loss = loss_fn(outputs.contiguous().view(-1, model.vocab_size), tt)

    loss.backward()

    hidden_grads = torch.stack(model.hidden_states)
    return hidden_grads.mean(1).norm(p=2, dim=-1)

if __name__ == "__main__":
    for model_type in ['RNN', 'GRU']:
        model = get_best_model(model_type)
        model = model.to(device)

        hidden_grads = run_batch(model, train_data)

        normalized = (hidden_grads - hidden_grads.min()) / (hidden_grads.max() - hidden_grads.min())

        plt.plot(range(model.seq_len), normalized.detach().cpu().numpy(), label=model_type)

    plt.title(f'Normalized norm of average hidden state gradients at each time-step')
    plt.legend()
    plt.xlabel('Time-step')
    plt.ylabel('Normalizaed norm of average hidden state gradient')
    plt.savefig(f'Q5_2_grad_wrt_time_steps.jpg')