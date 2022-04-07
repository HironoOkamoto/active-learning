import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import time
from os.path import join, exists, splitext, basename
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch
from torch.optim import lr_scheduler

from utils.utils import BaseExp
import timm

plt.style.use('ggplot')


class Exp(BaseExp):
    def __init__(self, args):
        super().__init__(args)
        self.NCLASS = 10
        self.log_dict ={"train": np.zeros((self.epochs, 2)), "test": np.zeros((self.epochs, 2))}


    def exp(self):
        transform_dict = {
                'train': transforms.Compose(
                    [transforms.Resize((224, 224)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]),
                     ]),
                'test': transforms.Compose(
                    [transforms.Resize((224, 224)),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]),
                     ])}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataloader = {}
        data_size = {}
        for phase in ["train", "test"]:
            dataset = torchvision.datasets.CIFAR10(root="data", train=phase=="train", download=True,
                                                   transform=transform_dict[phase])
            dataloader[phase] = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=phase=="train", num_workers=4, pin_memory=True)
            data_size[phase] = len(dataset.targets)

        model = timm.create_model(self.model, pretrained=True, num_classes=self.NCLASS)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=[int(0.8 * self.epochs)],
                                             gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        for epoch in range(self.epochs):
            print(f"epoch: {epoch+1}")
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                epoch_loss = 0.0
                running_corrects = 0

                for xs, ys in tqdm(dataloader[phase]):
                    xs = xs.to(device)
                    ys = ys.to(device)

                    if phase == "train":
                        outputs = model(xs)
                        loss = criterion(outputs, ys)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    else:
                        with torch.no_grad():
                            outputs = model(xs)
                            loss = criterion(outputs, ys)

                    _, preds = torch.max(outputs.data, 1)

                    epoch_loss += loss.item()
                    running_corrects += torch.sum(preds == ys)

                epoch_acc = running_corrects.item() / data_size[phase]
                epoch_loss /= data_size[phase]

                self.log_dict[phase][epoch, 0] = epoch_loss
                self.log_dict[phase][epoch, 1] = epoch_acc

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if self.log_dict["test"][epoch, 1] == self.log_dict["test"][:epoch + 1, 1].max():
                self.best_epoch = epoch
                self.best_acc = self.log_dict[phase][epoch, 1]
                torch.save(model.state_dict(), f'{self.exp_path}/model.pth')
            self.save_logs(epoch)
            scheduler.step()

        time_elapsed = time.time() - start_time
        print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print(f'Best val acc: {self.best_acc:.4f}')


    def save_logs(self, epoch):
        colors = {'train': 'tab:blue', 'test': 'tab:orange'}
        plt.clf()
        for phase in ["train", "test"]:
            plt.plot(self.log_dict[phase][:epoch + 1, 0], label=phase, color=colors[phase])
        plt.legend()
        plt.xlabel('epoch')
        plt.title('loss')
        plt.savefig(f'{self.exp_path}/loss.png')
        plt.close()

        plt.clf()
        for phase in ["train", "test"]:
            plt.plot(self.log_dict[phase][:epoch + 1, 1], label=phase, color=colors[phase])
        xlabel = f'epoch (best epoch {self.best_epoch} eval acc {self.best_acc:.4f})'
        plt.legend()
        plt.xlabel(xlabel)
        plt.title('accuracy')
        plt.savefig(f'{self.exp_path}/accuracy.png')
        plt.close()

        self.cfg["acc"] = f'{self.best_acc:.3f}'
        with open(f'{self.exp_path}/cfg.json', 'w') as f:
            f.write(json.dumps(self.cfg, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expid', '-i', type=str, default='000000')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--static_path', type=str, default='static')
    parser.add_argument('--epochs', '-e', type=int, default=6)
    parser.add_argument('--batch_size', '-bs', type=int, default=8)
    parser.add_argument('--model', '-m', type=str, default='efficientnet_b0')
    parser.add_argument('--lr', type=float, default=1e-3)

    exp = Exp(parser.parse_args())
    exp.exp()
