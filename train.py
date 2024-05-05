
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import loss_and_acc
from model import MobileNetV2
from torchvision import datasets, models, transforms
import time
import os
import copy
import setting

class training():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = setting.Epoch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset_size = 0
        self.dataset_class_name = ''

    def preprare_dataset(self):
        data_transforms = {
            'train': transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                         transforms.RandomRotation(degrees=15),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.CenterCrop(size=224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                         ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=setting.Batch_size,
                                                      shuffle=True, num_workers=4)
                       for x in ['train', 'val']}
        self.dataset_size = {x: len(image_datasets[x]) for x in ['train', 'val']}
        self.dataset_class_name = image_datasets['train'].classes

        return dataloaders

    def run(self):
        data_loaders = self.preprare_dataset()
        mobileNet_model = MobileNetV2(50).to(self.device)

        optimizer = optim.SGD(mobileNet_model.parameters(),
                              lr=setting.Learning_rate,
                              momentum=setting.Optim_momentum,
                              weight_decay=setting.Optim_weightDecay)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        model = self.run_train(mobileNet_model, optimizer, scheduler, data_loaders)

        torch.save(model.state_dict(), './checkpoint/test_model.pt')
        print('[Completed] Model saved')

    def run_train(self, model, optimizer, scheduler, data_loaders):
        loss_train = torch.zeros(self.epochs)
        loss_test = torch.zeros(self.epochs)
        acc_train = torch.zeros(self.epochs)
        acc_test = torch.zeros(self.epochs)

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch, self.epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in data_loaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_size[phase]
                epoch_acc = running_corrects.double() / self.dataset_size[phase]

                if phase == 'train':
                    loss_train[epoch] = epoch_loss
                    acc_train[epoch] = epoch_acc
                elif phase == 'val':
                    loss_test[epoch] = epoch_loss
                    acc_test[epoch] = epoch_acc

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        loss_and_acc(acc_train, loss_train, 'Graph of Train', self.epochs)
        loss_and_acc(acc_test, loss_test, 'Graph of Test', self.epochs)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)
        return model
