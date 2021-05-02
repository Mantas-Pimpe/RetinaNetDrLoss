from torch.utils.data import DataLoader

import model
import torch
import eval as evaluation
from dataloader import OIDV6Dataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torchvision import transforms

if __name__ == '__main__':
    labels_folder = "data/multidata/train/labels/"
    images_folder = "data/multidata/train/"
    labels_validation_folder = "data/multidata/validation/labels/"
    images_validation_folder = "data/multidata/validation/"
    classes_file = "data/classes.txt"

    dataset_train = OIDV6Dataset(labels_folder, images_folder, classes_file,
                                  transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_validation = OIDV6Dataset(labels_validation_folder, images_validation_folder, classes_file,
                               transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_validation is not None:
        sampler_validation = AspectRatioBasedSampler(dataset_validation, batch_size=1, drop_last=False)
        dataloader_validation = DataLoader(dataset_validation, num_workers=3, collate_fn=collater, batch_sampler=sampler_validation)

    model_file = "target/train_model_final.pt"
    model = torch.load(model_file)
    retinanet = model


    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()


    # model = torch.load(model_file)
    mAP = evaluation.evaluate(dataset_validation, retinanet, save_path='./target/')
# oidv6 downloader en --type_data all --classes Tie Door Laptop --limit 1000
# oidv6 downloader en --dataset data --type_data all --classes Tie Door Laptop --limit 1000 --multi_classes --yes

