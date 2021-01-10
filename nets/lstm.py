import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

MAX_QUESTION_LEN = 20
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, UNKNOWN_TOKEN]


class LSTM_question(nn.Module):
	def __init__(self, word_vocab_size, word_embedding_dim,  hidden_dim, out_dim: int,
				 batch_size, num_layers=2, p_dropout = 0.3):
		super(LSTM_question, self).__init__()

		self.batch_size = batch_size

		self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim, padding_idx=0)

		# Implement BiLSTM module which is fed with word embeddings and outputs hidden representations
		self.encoder = nn.LSTM(input_size=word_embedding_dim, hidden_size=hidden_dim,
							   num_layers=num_layers, bidirectional=True, batch_first=True,dropout=p_dropout)


	def forward(self, question, question_len):
		# Embedding the question
		word_embs = self.word_embedding(question) #  [batch_size, seq_length, word_emb_dim] [1, 16, 100]

		# pack the embedded rep
		packed_input = pack_padded_sequence(word_embs, question_len, batch_first=True,enforce_sorted=False)

		# pass the packed rep through bi-lstm
		lstm_out, hidden = self.encoder(packed_input)

		# unpack the lstm result
		output, input_sizes = pad_packed_sequence(lstm_out,total_length=MAX_QUESTION_LEN, batch_first=True) # [batch_size, max_seq_length, word_emb_dim]

		return output