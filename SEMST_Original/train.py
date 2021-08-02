import warnings
warnings.simplefilter("ignore", UserWarning)
import os
from pathlib import Path
import numpy as np
import copy
import argparse
from PIL import Image, ImageFile
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.optim import Adam
import torch.utils.data as data
from torchvision.utils import save_image
from torchvision import transforms
from dataset import PreprocessDataset, get_loader
from model import Model


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

def train_transform():
    transform_list = [
        transforms.Resize(size=(136)),
        transforms.RandomCrop(128),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

def main():
    parser = argparse.ArgumentParser(description='Structure-emphasized Multimodal Style Transfer by CHEN CHEN')
    parser.add_argument('--batch_size', '-b', type=int, default=4,
                        help='number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(negative value indicate CPU)')
    parser.add_argument('--learning_rate', '-lr', type=int, default=1e-5,
                        help='learning rate for Adam')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot to generate image')
    parser.add_argument('--alpha', default=1,
                        help='fusion degree, should be a float or a list which length is n_cluster')
    parser.add_argument('--gamma', type=float, default=1,
                        help='weight of style loss')
    parser.add_argument('--train_content_dir', type=str, default='../content',
                        help='content images directory for train')
    parser.add_argument('--train_style_dir', type=str, default='../style',
                        help='style images directory for train')
    parser.add_argument('--test_content_dir', type=str, default='../content',
                        help='content images directory for test')
    parser.add_argument('--test_style_dir', type=str, default='../style',
                        help='style images directory for test')
    parser.add_argument('--save_dir', type=str, default='result',
                        help='save directory for result and loss')
    parser.add_argument('--reuse', default=None,
                        help='model state path to load for reuse')

    args = parser.parse_args()

    # create directory to save
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    loss_dir = f'{args.save_dir}/loss'
    model_state_dir = f'{args.save_dir}/model_state'
    image_dir = f'{args.save_dir}/image'

    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)
        os.mkdir(model_state_dir)
        os.mkdir(image_dir)

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'
        print(f'# CUDA unavailable')

    print(f'# Minibatch-size: {args.batch_size}')
    print(f'# epoch: {args.epoch}')
    print('')

    # prepare dataset and dataLoader
    content_tf = train_transform()
    style_tf = train_transform()

    train_dataset = FlatFolderDataset(args.train_content_dir, content_tf)
    test_dataset = FlatFolderDataset(args.test_content_dir, content_tf)

    data_length = len(train_dataset)
    print(f'Length of train image pairs: {data_length}')

    train_loader = iter(data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(train_dataset),
        num_workers=8))
    test_iter = iter(data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(test_dataset),
        num_workers=8))

    # set model and optimizer
    model = Model(alpha=args.alpha,
                  device=device,
                  pre_train=True)

    if args.reuse is not None:
        model.load_state_dict(torch.load(args.reuse, map_location=lambda storage, loc: storage))
        print(f'{args.reuse} loaded')

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    prev_model = copy.deepcopy(model)
    prev_optimizer = copy.deepcopy(optimizer)

    # start training
    loss_list = []
    for e in range(1, args.epoch + 1):
        print(f'Start {e} epoch')
        i = 1
        for content_path, style_path, content_tensor, style_tensor in tqdm(train_loader):
            loss = model(content_path, style_path, content_tensor, style_tensor, args.gamma)
            if torch.isnan(loss):
                model = prev_model
                optimizer = torch.optim.Adam(model.parameters())
                optimizer.load_state_dict(prev_optimizer.state_dict())
            else:
                prev_model = copy.deepcopy(model)
                prev_optimizer = copy.deepcopy(optimizer)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())

                # print(f'[{e}/total {args.epoch} epoch],[{i} /'
                #       f'total {round(data_length/args.batch_size)} iteration]: {loss.item()}')

                if i % args.snapshot_interval == 0:
                    content_path, style_path, content_tensor, style_tensor = next(test_iter)
                    content = content_tensor.to(device)
                    style = style_tensor.to(device)
                    with torch.no_grad():
                        out = model.generate(content_path, style_path, content_tensor, style_tensor)
                    res = torch.cat([content, style, out], dim=0)
                    res = res.to('cpu')
                    save_image(res, f'{image_dir}/{e}_epoch_{i}_iteration.png', nrow=args.batch_size)
                    torch.save(model.state_dict(), f'{model_state_dir}/{e}_epoch_{i}_iteration.pth')
                i += 1
        torch.save(model.state_dict(), f'{model_state_dir}/{e}_epoch.pth')
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.savefig(f'{loss_dir}/train_loss.png')
    with open(f'{loss_dir}/loss_log.txt', 'w') as f:
        for l in loss_list:
            f.write(f'{l}\n')
    print(f'Loss saved in {loss_dir}')


if __name__ == '__main__':
    main()
