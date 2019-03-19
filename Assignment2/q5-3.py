import torch



from q5 import id_2_word, word_to_id, get_best_model, vocab_size
from models import MultiHeadedAttention, GRU, RNN, one_hot_encoding

batch_size_used = 20
generated_seq_len = 10


print([id_2_word[i] for i in range(10)])

start_word = word_to_id["the"]
start_word = torch.LongTensor([start_word for _ in range(batch_size_used)])
# print(start_word)
# exit()

model: RNN = get_best_model("RNN")
generated_tokens = model.generate(start_word, model.init_hidden(), generated_seq_len)
generated_tokens = generated_tokens.numpy()
for i in range(generated_seq_len):
    sentence = " ".join(map(id_2_word.get, generated_tokens[:,i]))
    print(sentence)