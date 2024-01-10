import torch
import numpy as np
import os
import argparse
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torchinfo import summary
from datasets import ExpertDataset
from torch import nn, optim
from ignite.metrics import Precision, Recall, Loss, Accuracy
from tqdm import tqdm
from aqua_rl.control.DQN import DQNNetwork
from aqua_rl import hyperparams


# function to reset all metrics
def reset_metrics(metrics):
    for metric in metrics:
        metric.reset()
    return

# function to update all metrics with batch data
def update_metrics(metrics, outputs, labels):
    for metric in metrics:
        metric.update((outputs, labels))
    return

def train(args=None):
    root = args.dataset_root
    model_save_path = args.save_path
    batch_size = args.batch_size
    analyze = args.analyze
    epochs = args.epochs
    history = hyperparams.history_size_
    pitch_actions = hyperparams.pitch_action_space_
    yaw_actions = hyperparams.yaw_action_space_
    metrics = []
    best_f1 = -1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DQNNetwork(history, int(pitch_actions*yaw_actions))
    model.to(device)
    summary(model)
    dataset = ExpertDataset(root=root, history=history)
    counts = np.zeros(int(pitch_actions*yaw_actions))
    for i in range(dataset.__len__()):
        counts[int(dataset.__getitem__(i)["action"].detach().cpu().numpy())] += 1
    class_weights = [1/c for c in counts]
    weights = []
    for i in range(dataset.__len__()):
        weights.append(class_weights[int(dataset.__getitem__(i)["action"].detach().cpu().numpy())])
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    # define loss function and optimizer. flexible to change
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # define metrics
    p = Precision(average=False, device=device)
    r = Recall(average=False, device=device)
    f1 = (p * r * 2 / (p + r)).mean()  # f1
    loss_metric = Loss(loss_fn, device=device)
    accuracy = Accuracy(device=device)
    metrics.append(f1)
    metrics.append(loss_metric)
    metrics.append(accuracy)

    # analyze training batches
    if analyze:
        print('Analyzing first batch')
        weight_counts = np.zeros(int(pitch_actions*yaw_actions))
        for _, sample in enumerate(loader):
            action = sample['action']
            for a in action.detach().cpu().numpy():
                weight_counts[int(a)] += 1
            print("Weight counts in batch: ", weight_counts)
            break

    # training loop
    print("Starting training for {} epochs on {}".format(epochs, device))
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch, epochs))

        model.train()  # set model to training mode

        # reset metrics
        reset_metrics(metrics)

        for _, sample in enumerate(tqdm(loader)):
            state = sample['state'].to(device)
            depth = sample['depth'].to(device)
            action = sample['action'].to(device=device, dtype=torch.int64)
            optimizer.zero_grad()  # clear gradients
            outputs = model(state, depth)  # run model on inputs

            # calculate loss between predictions and ground truth
            loss = loss_fn(outputs, action)
            loss.backward()  # backpropagate loss
            optimizer.step()  # update weights

            # update metrics
            update_metrics(metrics, outputs, action)

        # get training statistics
        train_f1 = metrics[0].compute()
        train_loss = metrics[1].compute()
        train_accuracy = metrics[2].compute()
        print("Training Stats: Epoch {}, Loss: {:.4f}, F1: {:.4f}, Accuracy: {:.4f}".format(
            epoch, train_loss, train_f1, train_accuracy))      

        if train_f1 >= best_f1:
                print('Updating best model')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                }, os.path.join(model_save_path, 'best.pt'))
                best_f1 = train_f1 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='./pid_expert/', type=str,
                        help='path to dataset root')
    parser.add_argument('--save_path', default='./models/', type=str,
                        help='path to save the model')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size to use during training')
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of epochs to train for')
    parser.add_argument('--analyze', default=False, type=bool,
                        help='analyze images from batch')
    args = parser.parse_args()
    train(args)
