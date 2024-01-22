import torch
from torchvision import models
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchinfo import summary
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import SegmentationDataset
from torch import nn, optim
import numpy as np
from ignite.metrics import Precision, Recall, Loss, Accuracy
from tqdm import tqdm
import os
import argparse

import segmentation_models_pytorch as smp

# data transformations
data_transforms = {
    'train': A.Compose([A.Resize(
                320,
                416,
                always_apply=True,
            ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0,
                           rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]),
    'valid': A.Compose([A.Resize(
                320,
                416,
                always_apply=True,
            ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
}

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


def load_model(n_classes, encoder_name = None):
    if encoder_name is None:
       model = models.segmentation.deeplabv3_mobilenet_v3_large(
           pretrained=True, progress=True)
       # freeze weights
       for param in model.parameters():
           param.requires_grad = False
       # replace classifier
       model.classifier = DeepLabHead(960, num_classes=n_classes)
    else:
        model = smp.Unet(
            encoder_name = encoder_name,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights = "imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                    # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_classes,                # model output channels (number of classes in your dataset)
        )
    return model

def train(args=None):
    root = args.dataset_root
    model_save_path = args.save_path
    batch_size = args.batch_size
    analyze = args.analyze
    epochs = args.epochs
    encoder_name = args.encoder_name
    metrics = []
    best_f1 = -1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(2, encoder_name)
    model.to(device)
    summary(model)
    datasets = {x: SegmentationDataset(root=root,
                                   split=x, transforms=data_transforms[x]) for x in ['train', 'valid']}
    train_sampler = RandomSampler(datasets['train'])
    valid_sampler = SequentialSampler(datasets['valid'])
    train_loader = DataLoader(
        datasets['train'], sampler=train_sampler, batch_size=batch_size)
    valid_loader = DataLoader(
        datasets['valid'], sampler=valid_sampler, batch_size=batch_size)

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
        import cv2
        import time
        print('Analyzing first batch')
        for _, sample in enumerate(train_loader):
            features = sample['image']
            masks = sample['mask']
            # display all images in batch
            for i, feature in enumerate(features):
                concat = np.concatenate((feature.movedim(0, -1).detach().cpu().numpy(),
                                        np.stack((masks[i].detach().cpu().numpy() * 255,)*3, axis=-1)), axis=1)
                cv2.imshow('Sample', concat)
                cv2.waitKey(250)
                time.sleep(0.5)
            break

    # training loop
    print("Starting training for {} epochs on {}".format(epochs, device))
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch, epochs))

        model.train()  # set model to training mode

        # reset metrics
        reset_metrics(metrics)

        for _, sample in enumerate(tqdm(train_loader)):
            features = sample['image'].to(device)
            masks = sample['mask'].to(device=device, dtype=torch.int64)

            optimizer.zero_grad()  # clear gradients
            if encoder_name is None:
                outputs = model(features)['out']  # run model on inputs

            else:
                outputs = model(features)#['out']

            # #display outputs
            # for ind,out in enumerate(outputs):
            #     im = np.stack((torch.argmax(out, dim=0).detach().cpu().numpy() * 255,)*3, axis=-1).astype(np.uint8)
            #     mask = np.stack((masks[ind].detach().cpu().numpy() * 255,)*3, axis=-1).astype(np.uint8)
            #     concat = np.concatenate((im,mask), axis=1)
            #     cv2.imshow('Out', concat)
            #     cv2.waitKey(250)
            #     time.sleep(0.5)

            # calculate loss between predictions and ground truth
            loss = loss_fn(outputs, masks)
            loss.backward()  # backpropagate loss
            optimizer.step()  # update weights

            # update metrics
            update_metrics(metrics, outputs, masks)

        # get training statistics
        train_f1 = metrics[0].compute()
        train_loss = metrics[1].compute()
        train_accuracy = metrics[2].compute()
        print("Training Stats: Epoch {}, Loss: {:.4f}, F1: {:.4f}, Accuracy: {:.4f}".format(
            epoch, train_loss, train_f1, train_accuracy))

        # reset metrics
        reset_metrics(metrics)

        # evaluate model on validation set
        with torch.no_grad():  # dont calculate gradients for validation set
            model.eval()  # set model to eval mode
            for _, sample in enumerate(tqdm(valid_loader)):
                features = sample['image'].to(device)
                masks = sample['mask'].to(device=device, dtype=torch.int64)

                outputs = model(features)['out']  # run model on inputs
                # calculate loss between predictions and ground truth
                loss = loss_fn(outputs, masks)

                # update metrics
                update_metrics(metrics, outputs, masks)

            # get valid statistics
            valid_f1 = metrics[0].compute()
            valid_loss = metrics[1].compute()
            valid_accuracy = metrics[2].compute()
            print("Valid Stats: Epoch {}, Loss: {:.4f}, F1: {:.4f}, Accuracy: {:.4f}".format(
                epoch, valid_loss, valid_f1, valid_accuracy))
            
            if valid_f1 >= best_f1:
                print('Updating best model')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                }, os.path.join(model_save_path, 'best.pt'))
                best_f1 = valid_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='simulation_rope_dataset_v2/', type=str,
                        help='path to dataset root')
    parser.add_argument('--save_path', default='models/', type=str,
                        help='path to save the model')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size to use during training')
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of epochs to train for')
    parser.add_argument('--analyze', default=False, type=bool,
                        help='analyze images from batch')
    parser.add_argument('--encoder_name', default=None, type=str,
                        help='encoder name')
    args = parser.parse_args()
    train(args)
