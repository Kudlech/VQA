import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
MAX_QUESTION_LEN = 20
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>" # Optional: this is used to pad a batch of sentences in different lengths.
SPECIAL_TOKENS = [PAD_TOKEN, UNKNOWN_TOKEN]
import torch.nn.functional as F


class LSTM_question(nn.Module):
	def __init__(self, word_vocab_size, word_embedding_dim,  hidden_dim, out_dim: int,
				 batch_size, num_layers=2, p_dropout = 0.3):
		super(LSTM_question, self).__init__()
		self.batch_size = batch_size
		self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim, padding_idx=0)
		self.encoder = nn.LSTM(input_size=word_embedding_dim, hidden_size=hidden_dim,
							   num_layers=num_layers, bidirectional=True, batch_first=True,dropout=p_dropout) # Implement BiLSTM module which is fed with word embeddings and outputs hidden representations
		# self.attention = nn.MultiheadAttention(2 * hidden_dim, 1)
		# self.fc = nn.Linear(2 * hidden_dim, out_dim)

	def forward(self, question, question_len):
		word_embs = self.word_embedding(question) #  [batch_size, seq_length, word_emb_dim] [1, 16, 100]
		packed_input = pack_padded_sequence(word_embs, question_len, batch_first=True,enforce_sorted=False)
		lstm_out, hidden = self.encoder(packed_input)
		output, input_sizes = pad_packed_sequence(lstm_out,total_length=MAX_QUESTION_LEN, batch_first=True) # [batch_size, max_seq_length, word_emb_dim]
		# attn, attn_weights = self.attention(output.transpose(0, 1), output.transpose(0, 1), output.transpose(0, 1))
		# return self.fc(output[torch.arange(output.size(0), out=torch.LongTensor()), question_len - 1]) # TODO: Check dimensions
		return output


# class AttnDecoder(nn.Module):
# 	def __init__(self, hidden_dim, word_vocab_size,word_embeding_dim,word_vectors, dropout_p=0.1,
# 				 max_length=MAX_COMMENT_LEN, atten_hyper = 0.5):
# 		super(AttnDecoder, self).__init__()
#
# 		self.hidden_dim = hidden_dim
# 		self.word_vocab_size = word_vocab_size
# 		self.dropout_p = dropout_p
# 		self.max_length = max_length
# 		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 		self.word_vectors = word_vectors
# 		self.embedding = nn.Embedding.from_pretrained(word_vectors, freeze=False)
#
# 		self.attn = nn.Linear(self.hidden_dim * 4, self.max_length)
# 		self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
# 		self.dropout = nn.Dropout(self.dropout_p)
#
# 		self.sequential = nn.Sequential(
# 			nn.Linear(5 * hidden_dim, 100),
# 			nn.Tanh(),
# 			nn.Linear(100, 1)
# 		)
# 		self.loss = nn.BCELoss()
# 		self.sigmoid = nn.Sigmoid()
#
# 		self.gama = atten_hyper
#
# 	def forward(self, input, hidden, encoder_outputs, toxic_level,comment_len):
# 		embedded = self.embedding(input.to(self.device))
# 		embedded = self.dropout(embedded)
# 		weights = []
#
# 		hidden_biderct = torch.cat((hidden[0][0], hidden[0][1]), dim=1)
# 		for i in range(len(encoder_outputs[0])):
# 			weights.append(self.attn(torch.cat((hidden_biderct,encoder_outputs[:,i]), dim=1)))
#
# 		weights = torch.stack(weights)
# 		weights = weights.permute(1,0,2)
# 		normalized_weights = F.softmax(weights, 2)
#
# 		attn_applied = torch.bmm(normalized_weights, encoder_outputs)
#
# 		output = torch.cat((attn_applied, embedded), dim=2)
#
# 		pool = []#torch.Tensor().to(self.device).unsqueeze(0)
# 		for idx, comment_len in enumerate(comment_len):
# 			max_pool, _ = torch.max(output[idx,:comment_len],0)
# 			pool.append(torch.mean(output[idx,:comment_len],0)*0.5 + 0.5*max_pool)
# 		pool = torch.stack(pool)
#
# 		output = self.sequential(pool)
# 		output = self.sigmoid(output)
# 		loss_continues = self.loss(output.squeeze(), toxic_level.to(self.device))
# 		loss_binary = self.loss(output.squeeze(),torch.round((toxic_level.to(self.device))))
# 		loss = (1-self.gama)*loss_continues + self.gama*loss_binary
# 		return loss, output
#
