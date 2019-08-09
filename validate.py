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
    parser.add_argument('--nsml_checkpoint', type=str, default="1,2,3,4,5,6,7,8,9")
    parser.add_argument('--nsml_session', type=str, default="185,221,242,243")
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

    parser.add_argument('--class_weight_adding', type=str, default="0.8,0.0,0.2,0.0")
    parser.add_argument('--transform', type=str, default="5crop")  # default, 5crop, 10crop
    parser.add_argument('--loss_type', type=str,
                        default="multi_soft_margin,multi_soft_margin,multi_soft_margin,multi_soft_margin")
    # cross_entropy, bce, multi_soft_margin, multi_margin, focal_loss, kldiv
    parser.add_argument('--model', type=str,
                        default="Resnet18,Resnet18,Resnet18,Resnet18")  # Resnet18, Resnet152, efficientnet-b7, baseline

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    models = args.model.split(",")
    class_weight_addings = args.class_weight_adding.split(",")
    loss_types = args.loss_type.split(",")
    nsml_checkpoints = args.nsml_checkpoint.split(",")
    nsml_sessions = args.nsml_session.split(",")

    print(args)

    for i, m_name in enumerate(models):
        loss_type = loss_types[i]
        nsml_ss = nsml_sessions[i]
        if m_name == "Resnet18":
            model = Resnet18(args.output_size)
        elif m_name == "Resnet152":
            model = Resnet152(args.output_size)
        elif m_name == "baseline":
            model = Baseline(args.hidden_size, args.output_size)
        elif m_name.split("-")[0] == "efficientnet":
            model = EfficientNet.from_pretrained(m_name, args.output_size)
        else:
            raise Exception("model type is invalid : " + m_name)

        class_weights = None
        if float(class_weight_addings[i]) > 0:
            class_weights = torch.tensor(get_class_weights(float(class_weight_addings[i]))).cuda()

        if loss_type == "cross_entropy":
            criterion = nn.CrossEntropyLoss(class_weights)
        elif loss_type == "bce":
            criterion = torch.nn.BCEWithLogitsLoss(class_weights)
        elif loss_type == "multi_soft_margin":
            criterion = nn.MultiLabelSoftMarginLoss(class_weights)
        elif loss_type == "multi_margin":
            criterion = nn.MultiLabelMarginLoss()
        elif loss_type == "focal_loss":
            criterion = FocalLoss2d(weight=class_weights)
        elif loss_type == "kldiv":
            criterion = torch.nn.KLDivLoss()
        else:
            raise Exception("loss type is invalid : " + args.loss_type)

        model = model.to(device)

        # DONOTCHANGE: They are reserved for nsml
        bind_model(model, args)
        if args.pause:
            nsml.paused(scope=locals())

        model.eval()
        transform = None
        batch_size = (256 if m_name == "Resnet18" else 32)
        if args.transform == "5crop":
            transform = transforms.Compose([transforms.Resize((args.input_size, args.input_size)),
                                            transforms.FiveCrop(args.input_size), transforms.Lambda(
                    lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])
            batch_size //= 5
        elif args.transform == "10crop":
            transform = transforms.Compose([transforms.TenCrop(args.input_size), transforms.Lambda(
                lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])
            batch_size //= 10

        dataloader, val_dataloader = train_dataloader(args.input_size, batch_size,
                                                      args.num_workers,
                                                      transform=transform)

        for nsml_cp in nsml_checkpoints:
            # Warning: Do not load data before this line
            nsml.load(checkpoint=nsml_cp, session="team_13/airush1/" + nsml_ss)

            total_loss = 0.
            total_correct = 0.
            total_ranking_ap_score = 0.
            total_ranking_loss = 0.
            print(model.__class__.__name__)
            print(criterion.__class__.__name__)
            print("team_13/airush1/" + nsml_ss, nsml_cp)
            # eval!
            for batch_idx, (image, tags) in enumerate(val_dataloader):
                image = image.to(device)
                tags = tags.to(device)

                if args.transform in ["5crop", "10crop"]:
                    # In your test loop you can do the following:
                    bs, ncrops, c, h, w = image.size()
                    output = model(image.view(-1, c, h, w)).double()  # fuse batch size and ncrops
                    output = output.view(bs, ncrops, -1).mean(1)  # avg over crops
                else:
                    output = model(image).double()
                if loss_type == "cross_entropy":
                    loss = criterion(output, torch.argmax(tags, dim=1))
                else:
                    loss = criterion(output, tags)

                output_prob = output
                if loss_type == "cross_entropy":
                    loss = criterion(output, torch.argmax(tags, dim=1))
                elif loss_type == "focal_loss":
                    output = F.sigmoid(output)
                    loss = criterion(output, tags)
                else:
                    loss = criterion(output, tags)

                predict_vector = np.argmax(to_np(output_prob), axis=1)

                tags_np = to_np(tags)
                output_prob_np = to_np(output_prob)

                ranking_ap_score = label_ranking_average_precision_score(tags_np, output_prob_np)
                ranking_loss = label_ranking_loss(tags_np, output_prob_np)

                label_vector = np.argmax(to_np(tags), axis=1)
                bool_vector = predict_vector == label_vector
                accuracy = bool_vector.sum() / len(bool_vector)

                if batch_idx % args.log_interval == 0:
                    print(
                        'Val [{}] Batch {} / {}: Batch Loss {:2.4f} / Batch Acc {:2.4f} / Lank AP {:2.4f} / Lank Loss {:2.4f}'.format(
                            datetime.now().strftime('%Y/%m/%d %H:%M:%S'), batch_idx,
                            len(val_dataloader),
                            loss.item(),
                            accuracy, ranking_ap_score, ranking_loss))
                total_loss += loss.item()
                total_correct += bool_vector.sum()
                total_ranking_ap_score += ranking_ap_score
                total_ranking_loss += ranking_loss

            print(
                'Val [{}] Epoch {} / {}: Loss {:2.4f} / Epoch Acc {:2.4f} / Lank AP {:2.4f} / Lank Loss {:2.4f}'.format(
                    datetime.now().strftime('%Y/%m/%d %H:%M:%S'), nsml_cp,
                    args.epochs,
                    total_loss / float(len(val_dataloader.dataset)),
                    total_correct / float(len(val_dataloader.dataset)),
                    total_ranking_ap_score / float(len(val_dataloader.dataset)),
                    total_ranking_loss / float(len(val_dataloader.dataset)),
                ))

            prefix = "%s_%s_%s" % (m_name, loss_type, class_weight_addings[i])
            nsml.report(
                summary=True,
                step=int(nsml_cp),
                scope=locals(),
                **{
                    "%s_%s__Loss" % (prefix, ("team_13/airush1/" + nsml_ss)): total_loss / float(
                        len(val_dataloader.dataset)),
                    "%s_%s__Accuracy" % (prefix, ("team_13/airush1/" + nsml_ss)): total_correct / float(
                        len(val_dataloader.dataset)),
                    "%s_%s__LankAp" % (prefix, ("team_13/airush1/" + nsml_ss)): total_ranking_ap_score / float(
                        len(val_dataloader.dataset)),
                    "%s_%s__LankLoss" % (prefix, ("team_13/airush1/" + nsml_ss)): total_ranking_loss / float(
                        len(val_dataloader.dataset)),
                })
