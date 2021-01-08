import torch
import hydra
import sys
from train import train, evaluate
from dataset import VQADataset
from models.base_model import MyModel
from torch.utils.data import DataLoader
from utils import main_utils, train_utils
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf
from tools.calc_std_mean import calc_std_mean
from compute_softscore import load_v2
import os

@hydra.main(config_path="config", config_name='config')
def	evaluate_hw2(cfg: DictConfig) -> float:
	# load_v2()

	# Load dataset

	path_image_train = '/datashare/train2014/COCO_train2014_'
	path_question_train = '/datashare/v2_OpenEnded_mscoco_train2014_questions.json'
	train_dataset = VQADataset(path_answers=rel_to_abs_path(cfg['main']['paths']['train']),
							   path_image=path_image_train, path_questions=path_question_train)
	path_image_val = '/datashare/val2014/COCO_val2014_'
	path_question_train = '/datashare/v2_OpenEnded_mscoco_val2014_questions.json'
	val_dataset = VQADataset(path_answers=rel_to_abs_path(cfg['main']['paths']['validation']), path_image=path_image_val,
							 path_questions=path_question_train, word_dict=train_dataset.word_dict)

	eval_loader = DataLoader(val_dataset, cfg['train']['batch_size'], shuffle=True,
							 num_workers=cfg['main']['num_workers'])

	image_dim = train_dataset.pic_size
	output_dim = 2410

	model = MyModel(batch_size=cfg['train']['batch_size'], word_vocab_size=train_dataset.vocab_size,
                    lstm_hidden=cfg['train']['num_hid'], output_dim=output_dim, dropout=cfg['train']['dropout'],
                    word_embedding_dim=cfg['train']['word_embedding_dim'], question_output_dim = cfg['train']['question_output_dim'],
                    image_dim= image_dim, last_hidden_fc_dim= cfg['train']['last_hidden_fc_dim'])
	if torch.cuda.is_available():
		model = model.cuda()
	model.load_state_dict(torch.load(rel_to_abs_path('model.pth'),map_location=lambda storage, loc: storage)['model_state'])
	model.train(False)
	eval_score, eval_loss = evaluate(model, eval_loader)

	print(f"The evaluation score is {eval_score %3}")

	return eval_score


def rel_to_abs_path(path):
	script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
	return os.path.join(script_dir, path)

if __name__ == '__main__':
	evaluate_hw2()