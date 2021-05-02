import argparse
import collections

import numpy as np
import torch
import torch.optim as optim
import model
import eval as evaluation

from torchvision import transforms
from dataloader import OIDV6Dataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader

# LSP: 1813010
# Model: RetinaNet
# Add. Loss Func.: DR loss
# Data classes: Tie, Door, Laptop



print('CUDA available: {}'.format(torch.cuda.is_available()))

use_gpu = True
labels_folder = "data/multidata/train/labels/"
images_folder = "data/multidata/train/"
labels_validation_folder = "data/multidata/validation/labels/"
images_validation_folder = "data/multidata/validation/"
classes_file = "data/classes.txt"

# labels_folder = "OIDv6/train/test_laptop/labels/"
# images_folder = "OIDv6/train/test_laptop/"
# labels_validation_folder = "OIDv6/validation/test_laptop/labels/"
# images_validation_folder = "OIDv6/validation/test_laptop/"
# classes_file = "OIDv6/classes.txt"
#
#
# labels_folder = "OIDv6/train/laptop/labels/"
# images_folder = "OIDv6/train/laptop/"
# labels_validation_folder = "OIDv6/validation/laptop/labels/"
# images_validation_folder = "OIDv6/validation/laptop/"
# classes_file = "OIDv6/classes.txt"


# Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50
depth = 50

# Number of epochs', type=int
epochs = 20

# Used for pt file (exmpl. laptop_retinanet_*epoch_nb*)
class_name = "./target/train"

def main(args=None):

    # Create the data loaders

    dataset_train = OIDV6Dataset(labels_folder, images_folder, classes_file,
                                  transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_validation = OIDV6Dataset(labels_validation_folder, images_validation_folder, classes_file,
                               transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_validation is not None:
        sampler_validation = AspectRatioBasedSampler(dataset_validation, batch_size=1, drop_last=False)
        dataloader_validation = DataLoader(dataset_validation, num_workers=3, collate_fn=collater, batch_sampler=sampler_validation)

    # Create the model
    if depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'
                    .format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        scheduler.step(np.mean(epoch_loss))

        print('Evaluating dataset')

        mAP = evaluation.evaluate(dataset_validation, retinanet, save_path='./target/')
        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(class_name, epoch_num))

    retinanet.eval()
    torch.save(retinanet, '{}_model_final.pt'.format(class_name))


if __name__ == '__main__':
    main()
