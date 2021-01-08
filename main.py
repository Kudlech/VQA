"""
Main file
We will run the whole program from here
"""

import torch
import hydra
import sys
from train import train
from dataset import VQADataset
from models.base_model import MyModel
from torch.utils.data import DataLoader
from utils import main_utils, train_utils
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf
from tools.calc_std_mean import calc_std_mean

torch.backends.cudnn.benchmark = True


@hydra.main(config_path="config", config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """
    main_utils.init(cfg)
    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    logger.write(OmegaConf.to_yaml(cfg))

    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])

    # Load dataset
    path_image_train = '/datashare/train2014/COCO_train2014_'
    path_question_train = '/datashare/v2_OpenEnded_mscoco_train2014_questions.json'
    train_dataset = VQADataset(path_answers=cfg['main']['paths']['train'],
                               path_image=path_image_train, path_questions=path_question_train)
    path_image_val = '/datashare/val2014/COCO_val2014_'
    path_question_val = '/datashare/v2_OpenEnded_mscoco_val2014_questions.json'
    val_dataset = VQADataset(path_answers=cfg['main']['paths']['validation'], path_image=path_image_val,
                             path_questions=path_question_val, word_dict=train_dataset.word_dict)

    train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['main']['num_workers'])
    eval_loader = DataLoader(val_dataset, cfg['train']['batch_size'], shuffle=True,
                             num_workers=cfg['main']['num_workers'])

    # std, mean = calc_std_mean(train_loader)
    # print(f"std:{std}, mean: {mean}")
    # Init model
    # batch_size, word_vocab_size, lstm_hidden = 256, output_dim = 2403, dropout = 0.2,
    # word_embedding_dim = 100, question_output_dim = 100, image_dim = 224 * 224, image_out_dim = 512):
    image_dim = train_dataset.pic_size
    output_dim =2410
    model = MyModel(batch_size=cfg['train']['batch_size'], word_vocab_size=train_dataset.vocab_size,
                    lstm_hidden=cfg['train']['num_hid'], output_dim=output_dim, dropout=cfg['train']['dropout'],
                    word_embedding_dim=cfg['train']['word_embedding_dim'], question_output_dim = cfg['train']['question_output_dim'],
                    image_dim= image_dim, last_hidden_fc_dim= cfg['train']['last_hidden_fc_dim'])

    # TODO: Add gpus_to_use
    if cfg['main']['parallel']:
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    logger.write(main_utils.get_model_string(model))

    # Run model
    train_params = train_utils.get_train_params(cfg)

    # Report metrics and hyper parameters to tensorboard
    metrics = train(model, train_loader, eval_loader, train_params, logger)
    hyper_parameters = main_utils.get_flatten_dict(cfg['train'])

    logger.report_metrics_hyper_params(hyper_parameters, metrics)


if __name__ == '__main__':
    main()
