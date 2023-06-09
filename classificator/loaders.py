import os
import time
import torch
import torchvision
import presets
import utils
from sampler import RASampler
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch.utils.data.dataloader import default_collate
from PIL import Image
import numpy as np
import transforms
import csv

def create_dataset_loader_samplers(train_dir, val_dir, args):

    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    one_batch = next(iter(data_loader))
    print(one_batch[0][0].shape)
    _, counts = np.unique(one_batch[1], return_counts=True)
    print('class distribution of one batch: ', dict(zip(dataset.classes, counts)))

    return dataset, dataset_test, data_loader, data_loader_test, \
           train_sampler, test_sampler

def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size, grayscale, mean, std = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
        args.grayscale,
        args.mean,
        args.std
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = args.ra_magnitude
        augmix_severity = args.augmix_severity

        if args.vitab:
            dataset = ViTabDataset(
                traindir + '/Images',
                traindir + '/particles_database.csv',
                presets.ClassificationPresetTrain(
                    mean=mean,
                    std=std,
                    crop_size=train_crop_size,
                    interpolation=interpolation,
                    auto_augment_policy=auto_augment_policy,
                    random_erase_prob=random_erase_prob,
                    ra_magnitude=ra_magnitude,
                    augmix_severity=augmix_severity,
                    grayscale=grayscale
                ),
            )

        else:
            dataset = torchvision.datasets.ImageFolder(
                traindir,
                presets.ClassificationPresetTrain(
                    mean=mean,
                    std=std,
                    crop_size=train_crop_size,
                    interpolation=interpolation,
                    auto_augment_policy=auto_augment_policy,
                    random_erase_prob=random_erase_prob,
                    ra_magnitude=ra_magnitude,
                    augmix_severity=augmix_severity,
                    grayscale=grayscale
                ),
            )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    ##################################
    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms()
        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size, resize_size=val_resize_size,
                interpolation=interpolation, grayscale=grayscale,
                mean=mean, std=std
            )
        if args.vitab:
            dataset_test = ViTabDataset(valdir + '/Images',
                valdir + '/particles_database.csv',
                preprocessing)
        else:
            dataset_test = torchvision.datasets.ImageFolder(
                valdir,
                preprocessing,
            )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    ###########################
    print("Creating data loaders")
    if False: #args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        if args.balance:
            _, class_counts = np.unique(dataset.targets, return_counts=True)
            weights = 1. / torch.tensor(class_counts, dtype=torch.float)
            print(class_counts)
            samples_weights = torch.from_numpy(
                np.array([weights[t] for t in dataset.targets]))

            train_sampler = WeightedRandomSampler(
                weights=samples_weights,
                num_samples=len(samples_weights),
                replacement=True)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)

        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

class ViTabDataset(Dataset):
    def __init__(self, image_paths, csv_file, transform=None):
        self.images_path = image_paths
        self.image_name = []
        self.tabular_data = []
        self.labels = []
        self.transform = transform

        # Read tabular data and labels from CSV file
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader, None)
            for row in csv_reader:
                self.image_name.append(row[0])
                self.labels.append(row[1])
                self.tabular_data.append(row[2:])

        self.labels = [label.replace(' ', '') for label in self.labels]
        self.classes = np.unique(self.labels)

        label_map = {label: i for i, label in enumerate(self.classes)}

        # Perform label encoding
        self.targets = torch.tensor(
            [label_map[label] for label in self.labels])


    def __getitem__(self, index):
        # Load image

        image = Image.open(os.path.join(self.images_path,
                                        self.image_name[index])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Get tabular features and label
        tabular_features = torch.tensor([np.float(feat)/100 for feat in self.tabular_data[index]])

        if self.labels:
            label = self.targets[index]
            data = (image, tabular_features)
            return data, label

        data = (image, tabular_features)
        return data

    def __len__(self):
        return len(self.image_name)


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

if __name__ == "__main__":

    path = "/media/lecun/HD/Expor2/ParticlesDB/csv/"
    img_path = path + "Images"
    csv_path = path + "particle_database.csv"
    data = load_data(img_path, csv_path)
    print(data)