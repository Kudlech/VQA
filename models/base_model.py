"""
    Example for a simple model
"""

from abc import ABCMeta
from nets.fc import FCNet
from torch import nn, Tensor
import torch
from nets.lstm import LSTM_question
from nets.image_net import ImageNet
class MyModel(nn.Module, metaclass=ABCMeta):
    """
    Example for a simple model
    """
    def __init__(self, batch_size, word_vocab_size, lstm_hidden=256, output_dim=2410, dropout=0.2,
                 word_embedding_dim =100, question_output_dim = 100, image_dim=224*224, pix_embd_dim = 32, attn_dim =16,
                 last_hidden_fc_dim=3000):
        super(MyModel, self).__init__()
        self.question_lstm = LSTM_question(word_vocab_size, word_embedding_dim, lstm_hidden, question_output_dim, batch_size,p_dropout=dropout)
        self.image_cnn = ImageNet(in_dim= image_dim, out_dim=pix_embd_dim, dropout=dropout)
        # self.attention = Attention(in_dim)
        self.fc_seq = nn.Sequential(
            nn.Linear(pix_embd_dim*49 + 20*40 + attn_dim*49 + attn_dim*20, last_hidden_fc_dim),
            nn.ReLU(),
            nn.Linear(last_hidden_fc_dim, output_dim)
        )
        self.attn_dim = attn_dim
        self.attn_image = nn.MultiheadAttention(pix_embd_dim, 2)
        self.attn_combine_image = nn.MultiheadAttention( self.attn_dim, 2)
        self.attn_combine_question = nn.MultiheadAttention( self.attn_dim, 2)
        self.attn_question = nn.MultiheadAttention(2*lstm_hidden, 2)
        self.linear_image = nn.Linear(pix_embd_dim,  self.attn_dim)
        self.linear_question = nn.Linear(2*lstm_hidden,  self.attn_dim)
    def forward(self, image, question,question_len) -> Tensor:
        """
        Forward x through MyModel
        :param x:
        :return:
        """
        batch_size = image.size(0)
        question_vec = self.question_lstm(question,question_len)
        question_vec = question_vec.transpose(0,1)
        image_vec = self.image_cnn(image)
        image_vec = image_vec.view(batch_size, 32, -1).permute(2, 0, 1) # [pixels, batch, image hidden_dim]
        self_attn_question_vec, _ = self.attn_question(question_vec, question_vec, question_vec, need_weights=False)
        self_attn_image_vec, _ = self.attn_image(image_vec,image_vec,image_vec, need_weights=False)
        linear_image = self.linear_image(image_vec) #[pixels, batch, shared hidden_dim]
        linear_question = self.linear_question(question_vec) #[sequence len, batch, shared hidden_dim]
        attn_combine_image, _ = self.attn_combine_image(linear_image, linear_question, linear_question, need_weights=False)
        attn_combine_question, _ = self.attn_combine_question(linear_question, linear_image, linear_image, need_weights=False)
        concat_vec = torch.cat((self_attn_image_vec.transpose(0,1).reshape(batch_size,-1),
                                self_attn_question_vec.transpose(0,1).reshape(batch_size,-1)
                                ,attn_combine_question.transpose(0,1).reshape(batch_size,-1),
                                attn_combine_image.transpose(0,1).reshape(batch_size,-1)
                                ), 1
                               )

        output = self.fc_seq(concat_vec)
        return output
