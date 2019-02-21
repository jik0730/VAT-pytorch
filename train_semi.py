import os
import json
import argparse
import logging
import torch
import torch.nn as nn
from tqdm import tqdm

import utils
from model import FFNN
from vat import VAT
from data_loader import fetch_dataloaders_MNIST

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument(
    '--model_dir',
    default='experiments/base_model',
    help="Directory containing params.json")


def train_single_iter(model, optimizer, loss_fn, reg_fn, dl_label, dl_unlabel,
                      params):
    model.train()

    label_X, label_y = dl_label.__iter__().next()
    unlabel_X, _ = dl_unlabel.__iter__().next()
    if params.cuda:
        label_X, label_y = label_X.cuda(async=True), label_y.cuda(async=True)
        unlabel_X = unlabel_X.cuda(async=True)

    label_logit = model(label_X)
    unlabel_logit = model(unlabel_X)
    nll = loss_fn(label_logit, label_y)
    vat = reg_fn(unlabel_X, unlabel_logit)
    loss = nll + vat
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return nll.item(), vat.item()


def evalutate(model, dl, params):
    model.eval()

    total, correct = 0, 0
    for test_X, test_y in dl:
        if params.cuda:
            test_X, test_y = test_X.cuda(async=True), test_y.cuda(async=True)
        logit = model(test_X)
        preds = torch.argmax(logit, dim=1)
        correct += torch.sum(preds == test_y).item()
        total += preds.size(0)
    return float(correct / total)


def train_and_evaluate(model, optimizer, scheduler, loss_fn, reg_fn,
                       dataloaders, params):
    dl_label = dataloaders['label']
    dl_unlabel = dataloaders['unlabel']
    dl_val = dataloaders['val']
    dl_test = dataloaders['test']

    # training steps
    is_best = False
    best_val_score = -float('inf')
    best_test_score = -float('inf')
    plot_history = {'val_acc': [], 'test_acc': []}
    for step in tqdm(range(params.n_iters)):
        if step >= params.decay_iter:
            scheduler.step()
        nll, vat = train_single_iter(model, optimizer, loss_fn, reg_fn,
                                     dl_label, dl_unlabel, params)
        # report logs for each iter (mini-batch)
        logging.info(
            "Iteration {}/{} ; LOSS {:05.3f} ; NLL {:05.3f} ; VAT {:05.3f}".
            format(step + 1, params.n_iters, nll + vat, nll, vat))
        if (step + 1) % params.n_summary_steps == 0:
            val_score = evalutate(model, dl_val, params)
            test_score = evalutate(model, dl_test, params)
            plot_history['val_acc'].append(val_score)
            plot_history['test_acc'].append(test_score)
            logging.info("Val_score {:05.3f} ; Test_score {:05.3f}".format(
                val_score, test_score))
            is_best = val_score > best_val_score
            if is_best:
                best_val_score = val_score
                best_test_score = test_score
                logging.info("Found new best accuray")
            print('[{}] Val score was {}'.format(step + 1, val_score))
            print('[{}] Test score was {}'.format(step + 1, test_score))
    print('Best val score was {}'.format(best_val_score))
    print('Best test score was {}'.format(best_test_score))

    # Store results
    results = {
        'Best val score': best_val_score,
        'Best test score': best_test_score
    }
    utils.save_dict_to_json(results,
                            os.path.join(args.model_dir, 'results.json'))
    utils.plot_training_results(args.model_dir, plot_history)


if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(params.SEED)
    if params.cuda: torch.cuda.manual_seed(params.SEED)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Define the model and optimizer
    if params.cuda:
        model = FFNN(params).cuda()
    else:
        model = FFNN(params)
    # TODO learning rate decay linearly
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, params.decay_step_size, params.decay_gamma)

    # fetch loss function and metrics
    loss_fn = nn.CrossEntropyLoss()
    # define reg_fn
    reg_fn = VAT(model, params)

    # fetch MNIST dataloaders
    dataloaders = fetch_dataloaders_MNIST(args.data_dir, params)

    # Train the model
    train_and_evaluate(model, optimizer, scheduler, loss_fn, reg_fn,
                       dataloaders, params)
