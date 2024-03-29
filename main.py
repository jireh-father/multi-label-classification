import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import pathlib
from model import Baseline, Resnet18, Resnet152, Resnext101, WideResnet101
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


# import adabound


def to_np(t):
    return t.cpu().detach().numpy()


def bind_model(model_nsml, args, infer_transform_list, infer_batch_size):
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

        batch_size = infer_batch_size  # you can change this. But when you use 'nsml submit --test' for test infer, there are only 200 number of data.
        device = 0

        dataloader = DataLoader(
            AIRushDataset(test_image_data_path, test_meta_data, label_path=None,
                          transform=transforms.Compose(
                              infer_transform_list)),
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


batch_size_map = {
    'Resnet18': 128,
    'Resnet152': 128,
    'efficientnet-b7': 16,
    'Resnext101': 64,
    'WideResnet101': 64
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parser')
    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)

    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu_num', type=int, nargs='+', default=[0])
    parser.add_argument('--output_size', type=int, default=350)  # Fixed

    # custom args
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--infer_batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)


    # train
    parser.add_argument('--load_nsml_cp', type=bool, default=True)
    parser.add_argument('--only_save', type=bool, default=False)
    parser.add_argument('--use_train', type=bool, default=True)
    parser.add_argument('--use_val', type=bool, default=False)
    parser.add_argument('--val_ratio', type=float, default=0.0)

    parser.add_argument('--transform_random_crop', type=bool, default=False)
    parser.add_argument('--transform_random_sized_crop', type=bool, default=True)
    parser.add_argument('--transform_norm', type=bool, default=False)
    parser.add_argument('--transform_hor_flip', type=bool, default=False)
    parser.add_argument('--transform_color_jitter', type=bool, default=False)

    parser.add_argument('--infer_transform_5crop', type=bool, default=False)
    parser.add_argument('--infer_transform_10crop', type=bool, default=False)
    parser.add_argument('--infer_transform_center_crop', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=40)  # 42)
    parser.add_argument('--sava_step_ratio', type=float, default=0.26)

    parser.add_argument('--learning_rate', type=float, default=0.0000015)  # 2.5e-4)
    parser.add_argument('--nsml_checkpoint', type=str, default="5")
    parser.add_argument('--nsml_session', type=str, default="team_13/airush1/385")

    parser.add_argument('--class_weight_adding', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)#0.00001)  # 0.00004)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--use_random_label', type=bool, default=False)

    parser.add_argument('--optimizer', type=str, default="sgd")  # adam, sgd, adabound, adamw
    parser.add_argument('--loss_type', type=str,
                        default="cross_entropy")  # cross_entropy, bce, multi_soft_margin, multi_margin, focal_loss, kldiv
    parser.add_argument('--model', type=str,
                        default="Resnet152")  # Resnet18, Resnet152, Resnext101, efficientnet-b7, WideResnet101, baseline

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    batch_size = batch_size_map[args.model]
    infer_batch_size = batch_size_map[args.model] // 2
    print("batch_size", batch_size)
    if args.model == "Resnet18":
        model = Resnet18(args.output_size)
    elif args.model == "Resnet152":
        model = Resnet152(args.output_size)
    elif args.model == "Resnext101":
        model = Resnext101(args.output_size)
    elif args.model == "baseline":
        model = Baseline(args.hidden_size, args.output_size)
    elif args.model == "WideResnet101":
        model = WideResnet101(args.output_size)
    elif args.model.split("-")[0] == "efficientnet":
        model = EfficientNet.from_pretrained(args.model, args.output_size)
    else:
        raise Exception("model type is invalid : " + args.model)

    if args.mode == "train":
        if args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9,
                                  weight_decay=args.weight_decay,
                                  nesterov=args.nesterov)
        # elif args.optimizer == "adabound":
        #     optimizer = adabound.AdaBound(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == "adamw":
            optimizer = optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer is invalid : " + args.optimizer)

        class_weights = None
        if args.class_weight_adding > 0:
            class_weights = torch.tensor(get_class_weights(args.class_weight_adding)).cuda()

        if args.loss_type == "cross_entropy":
            criterion = nn.CrossEntropyLoss(class_weights)
        elif args.loss_type == "bce":
            criterion = torch.nn.BCEWithLogitsLoss(class_weights)
        elif args.loss_type == "multi_soft_margin":
            criterion = nn.MultiLabelSoftMarginLoss(class_weights)
        elif args.loss_type == "multi_margin":
            criterion = nn.MultiLabelMarginLoss()
        elif args.loss_type == "focal_loss":
            criterion = FocalLoss2d(weight=class_weights)
        elif args.loss_type == "kldiv":
            criterion = torch.nn.KLDivLoss()
        else:
            raise Exception("loss type is invalid : " + args.loss_type)
        print(criterion.__class__.__name__)
        print(optimizer.__class__.__name__)

    print(model.__class__.__name__)
    print(args)

    transform_list = []
    infer_transform_list = []
    if args.transform_random_crop:
        transform_list.append(transforms.Resize((256, 256)))
        transform_list.append(transforms.RandomCrop((args.input_size, args.input_size)))
    elif args.transform_random_sized_crop:
        transform_list.append(transforms.RandomResizedCrop((args.input_size, args.input_size)))
    else:
        transform_list.append(transforms.Resize((args.input_size, args.input_size)))

    if args.transform_hor_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    if args.transform_color_jitter:
        transform_list.append(transforms.ColorJitter())

    transform_list.append(transforms.ToTensor())
    if args.transform_norm:
        transform_list.append(
            transforms.Normalize([0.44097832, 0.44847423, 0.42528335], [0.25748107, 0.26744914, 0.30532702]))

    if args.infer_transform_5crop or args.infer_transform_10crop:
        if args.transform_random_crop:
            infer_transform_list.append(transforms.Resize((248, 248)))
        elif args.transform_random_sized_crop:
            infer_transform_list.append(transforms.Resize((248, 248)))
        else:
            infer_transform_list.append(transforms.Resize((args.input_size, args.input_size)))

        if args.transform_hor_flip:
            infer_transform_list.append(transforms.RandomHorizontalFlip())
        if args.transform_color_jitter:
            infer_transform_list.append(transforms.ColorJitter())

        if args.infer_transform_5crop:
            infer_transform_list.append(transforms.FiveCrop((args.input_size, args.input_size)))
        else:
            infer_transform_list.append(transforms.TenCrop((args.input_size, args.input_size)))

        if args.transform_norm:
            infer_transform_list.append(transforms.Lambda(
                lambda crops: torch.stack([transforms.Normalize([0.44097832, 0.44847423, 0.42528335],
                                                                [0.25748107, 0.26744914, 0.30532702])(
                    transforms.ToTensor()(crop)) for crop in crops])))
        else:
            infer_transform_list.append(transforms.Lambda(
                lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if args.infer_transform_5crop:
            infer_batch_size = infer_batch_size // 5
        else:
            infer_batch_size = infer_batch_size // 10
    elif args.infer_transform_center_crop:
        infer_transform_list.append(transforms.Resize((248, 248)))
        infer_transform_list.append(transforms.CenterCrop((args.input_size, args.input_size)))
        infer_transform_list.append(transforms.ToTensor())
        if args.transform_norm:
            infer_transform_list.append(
                transforms.Normalize([0.44097832, 0.44847423, 0.42528335], [0.25748107, 0.26744914, 0.30532702]))
    else:
        if args.transform_random_crop:
            infer_transform_list.append(transforms.Resize((248, 248)))
            infer_transform_list.append(transforms.CenterCrop((args.input_size, args.input_size)))
        elif args.transform_random_sized_crop:
            infer_transform_list.append(transforms.Resize((248, 248)))
            infer_transform_list.append(transforms.CenterCrop((args.input_size, args.input_size)))
        else:
            infer_transform_list.append(transforms.Resize((args.input_size, args.input_size)))
        infer_transform_list.append(transforms.ToTensor())
        if args.transform_norm:
            infer_transform_list.append(
                transforms.Normalize([0.44097832, 0.44847423, 0.42528335], [0.25748107, 0.26744914, 0.30532702]))

    print("train transform", transform_list)
    print("val transform", infer_transform_list)
    model = model.to(device)

    # DONOTCHANGE: They are reserved for nsml
    bind_model(model, args, infer_transform_list, infer_batch_size)
    if args.pause:
        nsml.paused(scope=locals())
    if args.mode == "train":
        model.train()
        # Warning: Do not load data before this line
        epoch_start = 1

        if args.load_nsml_cp and args.nsml_checkpoint is not None and args.nsml_session is not None:
            nsml.load(checkpoint=args.nsml_checkpoint, session=args.nsml_session)
            print("load", args.nsml_session, args.nsml_checkpoint)
            if str.isnumeric(args.nsml_checkpoint):
                epoch_start += int(args.nsml_checkpoint)
                args.epochs += int(args.nsml_checkpoint)

        if args.only_save:
            nsml.save(args.nsml_session + "," + args.nsml_checkpoint)
        else:
            dataloader, val_dataloader = train_dataloader(args.input_size, batch_size, args.num_workers,
                                                          infer_batch_size=infer_batch_size,
                                                          transform=transforms.Compose(transform_list),
                                                          infer_transform=transforms.Compose(infer_transform_list),
                                                          val_ratio=args.val_ratio,
                                                          use_random_label=args.use_random_label, seed=args.seed)
            if args.sava_step_ratio > 0:
                save_step_interval = int(len(dataloader) * args.sava_step_ratio)
                print("save_step_interval", save_step_interval, "total steps", len(dataloader))
            else:
                save_step_interval = None

            for epoch_idx in range(epoch_start, args.epochs + 1):
                if args.use_train:
                    total_loss = 0.
                    total_correct = 0.
                    total_ranking_ap_score = 0.
                    total_ranking_loss = 0.
                    for batch_idx, (image, tags) in enumerate(dataloader):
                        optimizer.zero_grad()
                        image = image.to(device)
                        tags = tags.to(device)
                        output = model(image).double()
                        if args.loss_type == "cross_entropy":
                            loss = criterion(output, torch.argmax(tags, dim=1))
                        elif args.loss_type == "focal_loss":
                            output = F.softmax(output, dim=1)
                            loss = criterion(output, tags)
                        else:
                            loss = criterion(output, tags)
                        loss.backward()
                        optimizer.step()

                        output_prob = output
                        if args.loss_type == "cross_entropy":
                            output_prob = F.softmax(output, dim=1)
                        elif args.loss_type == "bce":
                            output_prob = torch.sigmoid(output)

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
                                'Train [{}] Batch {} / {}: Batch Loss {:2.4f} / Batch Acc {:2.4f} / Lank AP {:2.4f} / Lank Loss {:2.4f}'.format(
                                    datetime.now().strftime('%Y/%m/%d %H:%M:%S'), batch_idx,
                                    len(dataloader),
                                    loss.item(),
                                    accuracy, ranking_ap_score, ranking_loss))
                        total_loss += loss.item()
                        total_correct += bool_vector.sum()
                        total_ranking_ap_score += ranking_ap_score
                        total_ranking_loss += ranking_loss

                        if save_step_interval is not None and batch_idx > 0 and batch_idx % save_step_interval == 0:
                            nsml.save(str(epoch_idx) + "_" + str(batch_idx))

                    nsml.save(epoch_idx)
                    print(
                        'Train [{}] Epoch {} / {}: Loss {:2.4f} / Epoch Acc {:2.4f} / Lank AP {:2.4f} / Lank Loss {:2.4f}'.format(
                            datetime.now().strftime('%Y/%m/%d %H:%M:%S'), epoch_idx,
                            args.epochs,
                            total_loss / float(len(dataloader.dataset)),
                            total_correct / float(len(dataloader.dataset)),
                            total_ranking_ap_score / float(len(dataloader.dataset)),
                            total_ranking_loss / float(len(dataloader.dataset)),
                        ))
                    nsml.report(
                        summary=True,
                        step=epoch_idx,
                        scope=locals(),
                        **{
                            "train__Loss": total_loss / float(len(dataloader.dataset)),
                            "train__Accuracy": total_correct / float(len(dataloader.dataset)),
                            "train__LankAp": total_ranking_ap_score / float(len(dataloader.dataset)),
                            "train__LankLoss": total_ranking_loss / float(len(dataloader.dataset)),
                        })

                if args.use_val:
                    total_loss = 0.
                    total_correct = 0.
                    total_ranking_ap_score = 0.
                    total_ranking_loss = 0.

                    # eval!
                    model.eval()
                    for batch_idx, (image, tags) in enumerate(val_dataloader):
                        image = image.to(device)
                        tags = tags.to(device)

                        if args.infer_transform_5crop or args.infer_transform_10crop:
                            # In your test loop you can do the following:
                            bs, ncrops, c, h, w = image.size()
                            output = model(image.view(-1, c, h, w)).double()  # fuse batch size and ncrops
                            output = output.view(bs, ncrops, -1).mean(1)  # avg over crops
                        else:
                            output = model(image).double()

                        if args.loss_type == "cross_entropy":
                            loss = criterion(output, torch.argmax(tags, dim=1))
                        else:
                            loss = criterion(output, tags)

                        output_prob = output
                        if args.loss_type == "cross_entropy":
                            loss = criterion(output, torch.argmax(tags, dim=1))
                        elif args.loss_type == "focal_loss":
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
                            datetime.now().strftime('%Y/%m/%d %H:%M:%S'), epoch_idx,
                            args.epochs,
                            total_loss / float(len(val_dataloader.dataset)),
                            total_correct / float(len(val_dataloader.dataset)),
                            total_ranking_ap_score / float(len(val_dataloader.dataset)),
                            total_ranking_loss / float(len(val_dataloader.dataset)),
                        ))

                    nsml.report(
                        summary=True,
                        step=epoch_idx,
                        scope=locals(),
                        **{
                            "val__Loss": total_loss / float(len(val_dataloader.dataset)),
                            "val__Accuracy": total_correct / float(len(val_dataloader.dataset)),
                            "val__LankAp": total_ranking_ap_score / float(len(val_dataloader.dataset)),
                            "val__LankLoss": total_ranking_loss / float(len(val_dataloader.dataset)),
                        })
                    model.train()
                    if args.use_val and not args.use_train:
                        break
