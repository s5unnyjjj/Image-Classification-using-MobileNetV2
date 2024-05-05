
import torch
from torchvision import datasets, models, transforms
import csv
import os
from utils import TestDataSet

class evaluation():
    def __init__(self, path, model):
        self.test_dataset_path = path
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def run(self):
        transform_list = [transforms.Resize(size=256), transforms.CenterCrop(size=224),
                          transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

        _transforms = transforms.Compose(transform_list)
        test_dataset = TestDataSet(root=self.test_dataset_path, transform=_transforms)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=self.device)

        preds_list, image_names = self.run_eval(model=self.model, loader=test_dataloader, device=self.device)

        filename = 'result.csv'

        with open(filename, 'w', newline='') as csvfile:
            eval_writer = csv.writer(csvfile, delimiter=',')
            eval_writer.writerow(['ID', 'Category'])
            for i in range(len(preds_list)):
                eval_writer.writerow([i, int(preds_list[i])])

        print('[Completed] Evaluation')
        print('Results are saved at : ', os.path.join(os.getcwd(), filename))

    def run_eval(self, model, loader, device='cpu'):
        model.eval()
        preds_list = []
        image_name_list = []

        with torch.no_grad():
            for i, eval_data in enumerate(loader):
                eval_images, image_names = eval_data
                eval_images = eval_images.to(device)

                outputs = model(eval_images)
                preds = torch.argmax(outputs, 1)
                preds_list.extend(preds)
                image_name_list.extend(image_names)

        return preds_list, image_name_list


