"""
Here, we will run everything that is related to the training procedure.
"""

import time
import torch
import torch.nn as nn

from tqdm import tqdm
from utils import train_utils
from torch.utils.data import DataLoader
from utils.types import Scores, Metrics
from utils.train_utils import TrainParams
from utils.train_logger import TrainLogger
import sys

disable_tqdm = not(sys.stdin.isatty())

def get_metrics(best_eval_score: float, eval_score: float, train_loss: float) -> Metrics:
    """
    Example of metrics dictionary to be reported to tensorboard. Change it to your metrics
    :param best_eval_score:
    :param eval_score:
    :param train_loss:
    :return:
    """
    return {'Metrics/BestAccuracy': best_eval_score,
            'Metrics/LastAccuracy': eval_score,
            'Metrics/LastLoss': train_loss}


def train(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, train_params: TrainParams,
          logger: TrainLogger) -> Metrics:
    """
    Training procedure. Change each part if needed (optimizer, loss, etc.)
    :param model:
    :param train_loader:
    :param eval_loader:
    :param train_params:
    :param logger:
    :return:
    """
    metrics = train_utils.get_zeroed_metrics_dict()
    best_eval_score = 0

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_params.lr_step_size,
                                                gamma=train_params.lr_gamma)

    for epoch in (range(train_params.num_epochs)):
        print(f"#######epoch {epoch+1}##########")
        t = time.time()
        metrics = train_utils.get_zeroed_metrics_dict()
        bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

        for i, (image, question, question_len, label) in tqdm(enumerate(train_loader),
                                                              disable=disable_tqdm, total=len(train_loader)):
            if torch.cuda.is_available():
                image = image.cuda()
                question = question.cuda()
                question_len = question_len.cuda()
                label = label.cuda()

            y_hat = model(image, question,question_len)
            y_hat_probs = nn.functional.log_softmax(y_hat)
            # target_probs = nn.functional.softmax(label)
            loss = bce_loss(y_hat, label)

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            metrics['total_norm'] += nn.utils.clip_grad_norm_(model.parameters(), train_params.grad_clip)
            metrics['count_norm'] += 1

            # NOTE! This function compute scores correctly only for one hot encoding representation of the logits
            batch_score = train_utils.compute_soft_accuracy(y_hat_probs, label)
            metrics['train_score'] += batch_score.item()

            metrics['train_loss'] += loss.item()

            # Report model to tensorboard
            if epoch == 0 and i == 0:
                logger.report_graph(model, (image, question, question_len))
            1==1

        # Learning rate scheduler step
        scheduler.step()

        # Calculate metrics
        metrics['train_loss'] /= len(train_loader.dataset)

        metrics['train_score'] /= len(train_loader.dataset)
        metrics['train_score'] *= 100

        norm = metrics['total_norm'] / metrics['count_norm']

        model.train(False)
        metrics['eval_score'], metrics['eval_loss'] = evaluate(model, eval_loader)
        model.train(True)

        epoch_time = time.time() - t
        logger.write_epoch_statistics(epoch, epoch_time, metrics['train_loss'], norm,
                                      metrics['train_score'], metrics['eval_score'],metrics['eval_loss'])

        scalars = {'Accuracy/Train': metrics['train_score'],
                   'Accuracy/Validation': metrics['eval_score'],
                   'Loss/Train': metrics['train_loss'],
                   'Loss/Validation': metrics['eval_loss']}

        logger.report_scalars(scalars, epoch)

        if metrics['eval_score'] > best_eval_score:
            best_eval_score = metrics['eval_score']
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer)

    return get_metrics(best_eval_score, metrics['eval_score'], metrics['train_loss'])


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader) -> Scores:
    """
    Evaluate a model without gradient calculation
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :return: tuple of (accuracy, loss) values
    """
    score = 0
    bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
    loss = 0

    for i, (image, question, question_len, label) in tqdm(enumerate(dataloader),disable=disable_tqdm
            ,total=len(dataloader)):
        if torch.cuda.is_available():
            image = image.cuda()
            question = question.cuda()
            question_len = question_len.cuda()
            label = label.cuda()

        y_hat = model(image, question, question_len)
        y_hat_probs = nn.functional.log_softmax(y_hat)
        # target_probs = nn.functional.softmax(label)
        loss += bce_loss(y_hat, label)

        score += train_utils.compute_soft_accuracy(y_hat_probs, label).item()

    loss /= len(dataloader.dataset)
    score /= len(dataloader.dataset)
    score *= 100

    return score, loss
