import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import pathlib
from model import Baseline, Resnet18, Resnet152
import nsml
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataloader import train_dataloader, get_class_weights
from dataloader import AIRushDataset
from datetime import datetime
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import label_ranking_average_precision_score, label_ranking_loss
from loss import FocalLoss2d


def to_np(t):
    return t.cpu().detach().numpy()

batch_size_map = {
    'Resnet18': 128,
    'Resnet152': 128,
    'efficientnet-b7': 16,
}

def bind_model(model_nsml, args):
    def save(dir_name, **kwargs):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = {
            'model': model_nsml.state_dict(),
        }
        torch.save(state, save_state_path)
        print("model saved")

    def load(dir_name):
        save_state_path = os.path.join(dir_name, 'state_dict.pkl')
        state = torch.load(save_state_path)
        model_nsml.load_state_dict(state['model'])
        print("model loaded")

    def infer(test_image_data_path, test_meta_data_path):
        # DONOTCHANGE This Line
        test_meta_data = pd.read_csv(test_meta_data_path, delimiter=',', header=0)

        device = 0

        models = args.models.split(",")
        model_weights = [float(w) for w in args.model_weights.split(",")]
        nsml_sessionss = args.nsml_sessionss.split(",")
        nsml_checkpoints = args.nsml_checkpoints.split(",")
        loss_types = args.loss_types.split(",")

        total_output_probs = None
        before_batch_size = -1
        for i, model_name in enumerate(models):
            batch_size = batch_size_map[model_name] // 2
            if before_batch_size != batch_size:
                dataloader = DataLoader(
                    AIRushDataset(test_image_data_path, test_meta_data, label_path=None,
                                  transform=transforms.Compose(
                                      [transforms.Resize((args.input_size, args.input_size)), transforms.ToTensor()])),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True)

            before_batch_size = batch_size

            if model_name == "Resnet18":
                model = Resnet18(args.output_size)
            elif model_name == "Resnet152":
                model = Resnet152(args.output_size)
            elif model_name == "baseline":
                model = Baseline(args.hidden_size, args.output_size)
            elif model_name.split("-")[0] == "efficientnet":
                model = EfficientNet.from_pretrained(args.model, args.output_size)
            else:
                raise Exception("model type is invalid : " + args.model)

            model.to(device)

            def load_fn(dir_name):
                save_state_path = os.path.join(dir_name, 'state_dict.pkl')
                state = torch.load(save_state_path)
                model.load_state_dict(state['model'])
                print("model loaded", dir_name)


            model.eval()

            nsml.load(checkpoint=nsml_checkpoints[i], load_fn=load_fn, session="team_13/airush1/" + nsml_sessionss[i])

            output_probs = None
            for batch_idx, image in enumerate(dataloader):
                image = image.to(device)
                output = model(image).double()

                if loss_types[i] == "cross_entropy":
                    output_prob = F.softmax(output, dim=1)
                else:
                    output_prob = torch.sigmoid(output)

                if output_probs is None:
                    output_probs = to_np(output_prob)
                else:
                    output_probs = np.concatenate([output_probs, to_np(output_prob)], axis=0)
            if total_output_probs is None:
                total_output_probs = output_probs * model_weights[i]
            else:
                total_output_probs += (output_probs * model_weights[i])

        predict = np.argmax(total_output_probs, axis=1)

        return predict  # this return type should be a numpy array which has shape of (138343)

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_size', type=int, default=350)  # Fixed

    # custom args
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128)

    # train
    # cross_entropy, bce, multi_soft_margin, multi_margin, focal_loss, kldiv
    parser.add_argument('--loss_types', type=str, default="cross_entropy,cross_entropy")
    parser.add_argument('--nsml_checkpoints', type=str, default="3,4")
    parser.add_argument('--nsml_sessionss', type=str, default="99,99")  # team_13/airush1/
    parser.add_argument('--model_weights', type=str, default="0.5,0.5")
    parser.add_argument('--models', type=str,
                        default="Resnet152,Resnet152")  # Resnet18, Resnet152, efficientnet-b7, baseline

    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    device = args.device

    model = Resnet18(args.output_size)
    model = model.to(device)

    # DONOTCHANGE: They are reserved for nsml
    bind_model(model, args)
    if args.pause:
        nsml.paused(scope=locals())
    if args.mode == "train":
        model.train()
        nsml.save("ensemble_session")
