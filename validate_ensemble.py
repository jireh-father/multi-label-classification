import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import pathlib
from model import Baseline, Resnet18, Resnet152, Resnext101
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
    'Resnext101': 64,
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

        input_size = args.input_size  # you can change this according to your model.
        batch_size = args.infer_batch_size  # you can change this. But when you use 'nsml submit --test' for test infer, there are only 200 number of data.
        device = 0

        dataloader = DataLoader(
            AIRushDataset(test_image_data_path, test_meta_data, label_path=None,
                          transform=transforms.Compose(
                              [transforms.Resize((input_size, input_size)), transforms.ToTensor()])),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

        model_nsml.to(device)
        model_nsml.eval()
        predict_list = []
        for batch_idx, image in enumerate(dataloader):
            image = image.to(device)
            output = model_nsml(image).double()

            output_prob = output
            if args.loss_type == "cross_entropy":
                output_prob = F.softmax(output, dim=1)
            elif args.loss_type == "bce":
                output_prob = torch.sigmoid(output)

            predict = np.argmax(to_np(output_prob), axis=1)
            predict_list.append(predict)

        predict_vector = np.concatenate(predict_list, axis=0)
        return predict_vector  # this return type should be a numpy array which has shape of (138343)

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
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_num', type=int, nargs='+', default=[0])
    parser.add_argument('--output_size', type=int, default=350)  # Fixed

    # train
    parser.add_argument('--nsml_checkpoint', type=str, default="4,4")
    parser.add_argument('--nsml_session', type=str, default="220,243")
    parser.add_argument('--load_nsml_cp', type=bool, default=True)
    parser.add_argument('--only_save', type=bool, default=False)
    parser.add_argument('--use_train', type=bool, default=False)
    parser.add_argument('--use_val', type=bool, default=True)

    # custom args
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--infer_batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=2.5e-4)

    parser.add_argument('--model_weight', type=str, default="0.5,0.5")
    # cross_entropy, bce, multi_soft_margin, multi_margin, focal_loss, kldiv
    parser.add_argument('--model', type=str,
                        default="Resnet18,Resnet18")  # Resnet18, Resnet152, efficientnet-b7, baseline

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    models = args.model.split("/")
    nsml_checkpoints = args.nsml_checkpoint.split("/")
    nsml_sessions = args.nsml_session.split("/")

    model_weights = args.model_weight.split("/")

    print(args)

    for i, m_names in enumerate(models):
        m_name_list = m_names.split(",")
        nsml_checkpoint_list = nsml_checkpoints[i].split(",")
        nsml_session_list = nsml_sessions[i].split(",")
        model_weight_list = [float(w) for w in model_weights[i].split(",")]
        print(m_name_list)
        print(nsml_checkpoint_list)
        print("sessions", nsml_session_list)
        print(model_weight_list)
        total_output_probs = None
        total_tags = None
        for j, m_name in enumerate(m_name_list):
            nsml_ss = nsml_session_list[j]
            nsml_cp = nsml_checkpoint_list[j]
            model_weight = model_weight_list[j]
            if m_name == "Resnet18":
                model = Resnet18(args.output_size, False)
            elif m_name == "Resnet152":
                model = Resnet152(args.output_size, False)
            elif m_name == "baseline":
                model = Baseline(args.hidden_size, args.output_size)
            elif m_name.split("-")[0] == "efficientnet":
                model = EfficientNet.from_pretrained(m_name, args.output_s)
            elif args.model == "Resnext101":
                model = Resnext101(args.output_size, False)
            else:
                raise Exception("model type is invalid : " + m_name)

            model = model.to(device)

            # DONOTCHANGE: They are reserved for nsml
            bind_model(model, args)
            if args.pause:
                nsml.paused(scope=locals())

            model.eval()
            transform = None
            batch_size = batch_size_map[m_name] // 2

            dataloader, val_dataloader = train_dataloader(args.input_size, batch_size,
                                                          args.num_workers,
                                                          infer_batch_size=batch_size,
                                                          transform=transform,
                                                          infer_transform=transform)

            # Warning: Do not load data before this line
            nsml.load(checkpoint=nsml_cp, session="team_13/airush1/" + nsml_ss)

            print(model.__class__.__name__)
            print("team_13/airush1/" + nsml_ss, nsml_cp)
            # eval!
            output_probs = None

            for batch_idx, (image, tags) in enumerate(val_dataloader):
                image = image.to(device)
                if j == 0:
                    if total_tags is None:
                        total_tags = tags.detach().numpy()
                    else:
                        total_tags = np.concatenate([total_tags, tags.detach().numpy()], axis=0)

                output = model(image).double()

                # if loss_types[i] == "cross_entropy":
                #     output_prob = F.softmax(output, dim=1)
                # else:
                output_prob = torch.sigmoid(output)

                if output_probs is None:
                    output_probs = to_np(output_prob)
                else:
                    output_probs = np.concatenate([output_probs, to_np(output_prob)], axis=0)

            if total_output_probs is None:
                total_output_probs = output_probs * model_weight
            else:
                total_output_probs += (output_probs * model_weight)

        predict_vector = np.argmax(total_output_probs, axis=1)

        ranking_ap_score = label_ranking_average_precision_score(total_tags, total_output_probs)
        ranking_loss = label_ranking_loss(total_tags, total_output_probs)

        label_vector = np.argmax(total_tags, axis=1)
        bool_vector = predict_vector == label_vector
        accuracy = bool_vector.sum() / len(bool_vector)

        ens_ranking_ap_score = ranking_ap_score
        ens_ranking_loss = ranking_loss

        print(
            'Ens Val [{}] Acc {:2.4f} / Lank AP {:2.4f} / Lank Loss {:2.4f}'.format(
                datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
                accuracy, ens_ranking_ap_score, ens_ranking_loss))

        m_name_list = m_names.split(",")
        nsml_checkpoint_list = nsml_checkpoints[i].split(",")
        nsml_session_list = nsml_sessions[i].split(",")
        model_weight_list = [float(w) for w in model_weights[i].split(",")]

        prefix = "%s_%s_%s" % (
            nsml_checkpoint_list, m_name_list, model_weight_list)
        nsml.report(
            summary=True,
            step=i,
            scope=locals(),
            **{
                "ens_%s_%s__Accuracy" % (nsml_session_list, prefix): accuracy,
                "ens_%s_%s__LankAp" % (nsml_session_list, prefix): ens_ranking_ap_score,
                "ens_%s_%s__LankLoss" % (nsml_session_list, prefix): ens_ranking_loss,
            })
