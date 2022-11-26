import argparse
from sre_constants import GROUPREF_IGNORE
import time
import os
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from config import config_training_setup
from src.imageaugmentations import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip
from src.model_utils import load_network
from torch.utils.data import DataLoader
import cv2

def max_logits_loss(logits):
    return

def max_logits_attack(num_classes, network, x, _logits, delta_eps, target, ignore_train_ind):
    detect_ood = True
    logits = _logits.cpu()
    max_logits = torch.max(logits, dim=1).values
    max_class = torch.argmax(logits, dim=1)
    ignore_area = torch.isin(target, torch.Tensor(ignore_train_ind))
    if detect_ood == True:
        ignore_area = target==0 + ignore_area
    noise = torch.zeros_like(x)
    for c in range(num_classes):
        if c in max_class[~ignore_area]:
            mask = max_class==c * ~ignore_area
            avg_max_logits = torch.sum(torch.where(mask, max_logits, 0.)) / torch.sum(mask)
            avg_max_logits.backward(retain_graph=True)
            normed_grad = torch.sign(x.grad)
            stacked_max_class = torch.stack([max_class for _ in range(x.shape[1])], dim=1)
            stacked_ignore_area = torch.stack([ignore_area for _ in range(x.shape[1])], dim=1)
            normed_grad[stacked_max_class!=c] = 0
            normed_grad[stacked_ignore_area] = 0
            noise += delta_eps * normed_grad
            x.grad.data.zero_()
    lower_logits = network((x-noise).cuda()).cpu()
    lower_max_logits = torch.max(lower_logits, dim=1).values
    print(torch.sum(  (max_class!=torch.argmax(lower_logits, dim = 1))[~ignore_area]  ))
    # print(torch.sort(torch.unique(max_logits)).values)
    delta_max_logits = ((max_logits-lower_max_logits)/max_logits).detach().cpu()
    print("gap is: ", torch.sort(torch.unique((delta_max_logits[~ignore_area]))))
    # y_data = np.sort(np.reshape(delta_max_logits[~ignore_area].numpy(), -1))
    # fig = plt.figure
    # plt.plot(range(y_data.shape[0]), y_data)
    # if detect_ood:
    #     plt.savefig('./delta_distribution_{a}.png'.format(a='ood'))
    # else:
    #     plt.savefig('./delta_distribution_{a}.png'.format(a='id'))
    # print(torch.sum((max_logits-lower_max_logits)[~ignore_area]>0)/torch.sum((max_logits-lower_max_logits)[~ignore_area]!=0))
    return 

def cross_entropy(logits, targets):
    """
    cross entropy loss with one/all hot encoded targets -> logits.size()=targets.size()
    :param logits: torch tensor with logits obtained from network forward pass
    :param targets: torch tensor one/all hot encoded
    :return: computed loss
    """
    neg_log_like = - 1.0 * F.log_softmax(logits, 1)
    L = torch.mul(targets.float(), neg_log_like)
    L = L.mean()
    return L
# remember to modify shuffle
def cal_delta_entropy(network, x, delta_eps, delta_targets, original_targets, original_loss):
    grad = x.grad
    highloss_delta = x + delta_eps*grad
    highloss_logits = network(highloss_delta.cuda())
    high_entropy_loss = cross_entropy(highloss_logits, original_targets)
    print("gap is: ", ((high_entropy_loss-original_loss)/original_loss).item())
    performance_variance = -1 * cross_entropy(highloss_logits, delta_targets)
    return performance_variance

def encode_target(target, pareto_alpha, num_classes, ignore_train_ind, ood_ind=254):
    """
    encode target tensor with all hot encoding for OoD samples
    :param target: torch tensor
    :param pareto_alpha: OoD loss weight
    :param num_classes: number of classes in original task
    :param ignore_train_ind: void class in original task
    :param ood_ind: class label corresponding to OoD class
    :return: one/all hot encoded torch tensor
    """
    npy = target.numpy()
    npz = npy.copy()
    npy[np.isin(npy, ood_ind)] = num_classes
    npy[np.isin(npy, ignore_train_ind)] = num_classes + 1
    enc = np.eye(num_classes + 2)[npy][..., :-2]  # one hot encoding with last 2 axis cutoff
    enc[(npy == num_classes)] = np.full(num_classes, pareto_alpha / num_classes)  # set all hot encoded vector
    enc[(enc == 1)] = 1 - pareto_alpha  # convex combination between in and out distribution samples
    enc[np.isin(npz, ignore_train_ind)] = np.zeros(num_classes)
    enc_for_delta = enc.copy()
    enc_for_delta[(npy == num_classes)] = -1 * enc[(npy == num_classes)]
    enc = torch.from_numpy(enc)
    enc_for_delta = torch.from_numpy(enc_for_delta)
    enc = enc.permute(0, 3, 1, 2).contiguous()
    enc_for_delta = enc_for_delta.permute(0, 3, 1, 2).contiguous()
    return enc, enc_for_delta

def training_routine(config):
    """Start OoD Training"""
    print("START OOD TRAINING")
    params = config.params
    roots = config.roots
    dataset = config.dataset()
    print("Pareto alpha:", params.pareto_alpha)
    start_epoch = params.training_starting_epoch
    epochs = params.num_training_epochs
    start = time.time()

    """Initialize model"""
    if start_epoch == 0:
        network = load_network(model_name=roots.model_name, num_classes=dataset.num_classes,
                               ckpt_path=roots.init_ckpt, train=True)
    else:
        basename = roots.model_name + "_epoch_" + str(start_epoch) \
                   + "_alpha_" + str(params.pareto_alpha) + ".pth"
        network = load_network(model_name=roots.model_name, num_classes=dataset.num_classes,
                               ckpt_path=os.path.join(roots.weights_dir, basename), train=True)

    transform = Compose([RandomHorizontalFlip(), RandomCrop(params.crop_size), ToTensor(),
                         Normalize(dataset.mean, dataset.std)])

    for epoch in range(start_epoch, start_epoch + epochs):
        """Perform one epoch of training"""
        print('\nEpoch {}/{}'.format(epoch + 1, start_epoch + epochs))
        optimizer = optim.Adam(network.parameters(), lr=params.learning_rate)
        trainloader = config.dataset('train', transform, roots.cs_root, roots.coco_root, params.ood_subsampling_factor)
        dataloader = DataLoader(trainloader, batch_size=params.batch_size, shuffle=False, drop_last=True)
        i = 0
        loss = None
        for b_idx, (x, target) in enumerate(dataloader):
            x.requires_grad = True
            optimizer.zero_grad()
            # y, y_for_delta = encode_target(target=target, pareto_alpha=params.pareto_alpha, num_classes=dataset.num_classes,
            #                   ignore_train_ind=dataset.void_ind, ood_ind=dataset.train_id_out)
            # y, y_for_delta = y.cuda(), y_for_delta.cuda()
            logits = network(x.cuda())

            #print identity difference
            ignore_area = torch.isin(target, torch.Tensor(dataset.void_ind))
            another_logits = network(x.cuda())
            delta_max_logits = (torch.abs(torch.max(logits, dim=1).values-torch.max(another_logits, dim=1).values)/torch.max(logits, dim=1).values).detach()
                #print heat map
            delta_logits_pic = 255 * np.clip(np.abs(delta_max_logits.cpu().numpy()), 0, 1)
            cv2.imwrite("./heatmap.png", delta_logits_pic[0])
                #plot distribution
            y_data = np.sort(np.reshape(delta_max_logits[~ignore_area].cpu().numpy(), -1))
            fig = plt.figure
            plt.plot(range(y_data.shape[0]), y_data)
            plt.savefig('./delta_distribution_{a}.png'.format(a='identity'))
            exit()

            # loss = cross_entropy(logits, y)
            # loss.backward()
            # delta_entropy_loss = params.delta_rate * cal_delta_entropy(network, x, params.FGSM_eps, y_for_delta, y, loss)

            max_logits_attack(num_classes=dataset.num_classes, network=network, x=x, _logits=logits, 
                                delta_eps=params.FGSM_eps, target=target, ignore_train_ind=dataset.void_ind)

            x.requires_grad = False
            # optimizer.step()
            # print('{} Loss: {}, delta_Loss: {}'.format(i, loss.item(), delta_entropy_loss))
            i += 1
            if((b_idx + 1) % 100 == 0):
                """Save model state"""
                save_basename = roots.model_name + "_batch_" + str(b_idx + 1) + "_alpha_" + str(params.pareto_alpha) + ".pth"
                print('Saving checkpoint', os.path.join(roots.weights_dir, save_basename))
                torch.save({
                    'epoch': epoch + 1,
                    'batch': b_idx + 1,
                    'state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(roots.weights_dir, save_basename))
        torch.cuda.empty_cache()

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("FINISHED {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def main(args):
    """Perform training"""
    config = config_training_setup(args)
    training_routine(config)


if __name__ == '__main__':
    """Get Arguments and setup config class"""
    parser = argparse.ArgumentParser(description='OPTIONAL argument setting, see also config.py')
    parser.add_argument("-train", "--TRAINSET", nargs="?", type=str)
    parser.add_argument("-model", "--MODEL", nargs="?", type=str)
    parser.add_argument("-epoch", "--training_starting_epoch", nargs="?", type=int)
    parser.add_argument("-nepochs", "--num_training_epochs", nargs="?", type=int)
    parser.add_argument("-alpha", "--pareto_alpha", nargs="?", type=float)
    parser.add_argument("-lr", "--learning_rate", nargs="?", type=float)
    parser.add_argument("-crop", "--crop_size", nargs="?", type=int)
    main(vars(parser.parse_args()))
