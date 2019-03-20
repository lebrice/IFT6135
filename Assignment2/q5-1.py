import matplotlib.pyplot as plt
from q5 import get_best_model, run_epoch, valid_data, device

if __name__ == "__main__":
    for model_type in ['RNN', 'GRU', 'TRANSFORMER']:
        model = get_best_model(model_type)
        model = model.to(device)

        _, losses = run_epoch(model, valid_data)

        losses = losses.detach().cpu().numpy()
        plt.clf()
        plt.plot(range(len(losses)), losses)
        plt.title(f'Average loss at each time-step within validation sequences for {model_type} model')
        plt.xlabel('Time-step')
        plt.ylabel('Average loss')
        plt.savefig(f'Q5_1_loss_at_time_steps_{model_type}.jpg')