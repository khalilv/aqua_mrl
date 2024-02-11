import torch
from torchvision import models
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchinfo import summary
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import SegmentationDataset
import numpy as np
from ignite.metrics import Precision, Recall, Accuracy
from tqdm import tqdm
from time import time, sleep
import cv2
import argparse

# data transformations
data_transforms = {
    'test': A.Compose([A.Resize(
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


def load_model(n_classes):
    model = models.segmentation.deeplabv3_mobilenet_v3_large(
        pretrained=True, progress=True)
    # freeze weights
    for param in model.parameters():
        param.requires_grad = False
    # replace classifier
    model.classifier = DeepLabHead(960, num_classes=n_classes)
    return model


def eval(args=None):
    root = args.dataset_root
    checkpoint_path = args.checkpoint_path
    batch_size = args.batch_size
    average_fps = 0
    n = 0
    metrics = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(2)
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    summary(model)

    dataset = SegmentationDataset(root=root,
                              split='test', transforms=data_transforms['test'])
    sampler = SequentialSampler(dataset)
    loader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size)

    # define metrics
    p = Precision(average=False, device=device)
    r = Recall(average=False, device=device)
    f1 = (p * r * 2 / (p + r)).mean()  # f1
    accuracy = Accuracy(device=device)
    metrics.append(f1)
    metrics.append(accuracy)

    print("Starting evaluation on {}".format(device))

    reset_metrics(metrics)
    # evaluate model on test set
    with torch.no_grad():  # dont calculate gradients for validation set
        model.eval()  # set model to eval mode
        for _, sample in enumerate(tqdm(loader)):
            features = sample['image'].to(device)
            masks = sample['mask'].to(device=device, dtype=torch.int64)

            t0 = time()
            outputs = model(features)['out']  # run model on inputs
            t1 = time()
            if n == 0:
                average_fps = (t1 - t0)
            else:
                average_fps = average_fps + ((1/n) * ((t1 - t0) - average_fps))
            n += 1

            # # display outputs
            # for ind, out in enumerate(outputs):
            #     im = np.stack((torch.argmax(out, dim=0).detach(
            #     ).cpu().numpy() * 255,)*3, axis=-1).astype(np.uint8)
            #     mask = np.stack(
            #         (masks[ind].detach().cpu().numpy() * 255,)*3, axis=-1).astype(np.uint8)
            #     concat = np.concatenate((im, mask), axis=1)
            #     cv2.imshow('Out', concat)
            #     cv2.waitKey(250)
            #     sleep(0.5)

            # update metrics
            update_metrics(metrics, outputs, masks)

        # get valid statistics
        test_f1 = metrics[0].compute()
        test_accuracy = metrics[1].compute()
        print("Test Stats: F1: {:.4f}, Accuracy: {:.4f} FPS per batch: {:.4f}".format(
            test_f1, test_accuracy, average_fps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='simulation_dataset/', type=str,
                        help='path to dataset root')
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='trained model to load')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size to use during testing')
    args = parser.parse_args()
    eval(args)
