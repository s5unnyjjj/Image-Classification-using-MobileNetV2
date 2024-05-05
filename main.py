

from evaluate import evaluation
from train import training
import os
import setting


def start_training(dir):
    model = training(data_dir=dir).run()
    return model


def start_evaluation(dir, model):
    evaluation(path=os.path.join(dir, 'val'), model=model).run()


if __name__ == '__main__':
    base_dir = setting.Base_path

    model = start_training(dir=base_dir)
    print('[Completed] Training')

    start_evaluation(dir=base_dir, model=model)
    print('[Completed] Evaluation')
