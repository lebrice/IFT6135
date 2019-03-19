import torch



from q5 import id_2_word, word_to_id, get_best_model, vocab_size
from models import MultiHeadedAttention, GRU, RNN, one_hot_encoding

def generate_sequences(model_type: str, generated_seq_len: int, batch_size_used = 20, starting_word="<eos>"):
        
    starting_word = "<eos>"
    start_word = word_to_id[starting_word]
    start_word = torch.LongTensor([start_word for _ in range(batch_size_used)])

    model = get_best_model(model_type)
    initial_state = torch.zeros(model.num_layers, batch_size_used, model.hidden_size)
    generated_tokens = model.generate(start_word, initial_state, generated_seq_len)
    generated_tokens = generated_tokens.numpy()
    sentences = []
    for i in range(batch_size_used):
        sentence = starting_word + " "
        sentence += " ".join(map(id_2_word.get, generated_tokens[:,i]))
        sentences.append(sentence)
    return sentences

if __name__ == "__main__":
    training_seq_len = 35
    batch_size = 10
    from itertools import product
    import json
    results = {}
    for model_type, seq_len in product(("RNN", "GRU"), (training_seq_len, training_seq_len*2)):
        print(model_type, ":")
        sentences = generate_sequences(model_type, seq_len, batch_size)
        for sentence in sentences:
            print(sentence)
        results[f"{model_type}_{seq_len}"] = sentences

    with open("q5-3-results.json", "w") as file:
        json.dump(results, file, indent=1)