import os
import sys
import pickle
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy import stats
from src.model.deepv3 import DeepWV3Plus
from src.model.DualGCNNet import DualSeg_res50


def load_network(model_name, num_classes, ckpt_path=None, train=False):
    network = None
    print("Checkpoint file:", ckpt_path)
    print("Load model:", model_name, end="", flush=True)
    if model_name == "DeepLabV3+_WideResNet38":
        network = nn.DataParallel(DeepWV3Plus(num_classes))
    elif model_name == "DualGCNNet_res50":
        network = DualSeg_res50(num_classes)
    else:
        print("\nModel is not known")
        exit()

    if ckpt_path is not None:
        network.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
    network = network.cuda()
    if train:
        print("...train mode ok")
        return network.train()
    else:
        print("...eval mode ok")
        return network.eval()


def prediction(net, image):
    image = image.cuda()
    with torch.no_grad():
        out = net(image)
    if isinstance(out, tuple):
        out = out[0]
    out = out.data.cpu()
    out = F.softmax(out, 1)
    return out.numpy()

class inference(object):

    def __init__(self, params, roots, loader, num_classes=None, init_net=True, moment=False):
        self.epoch = params.val_epoch
        self.alpha = params.pareto_alpha
        self.batch_size = params.batch_size
        self.model_name = roots.model_name
        self.moment_oder = params.moment_order
        self.moment_num = params.moment_num
        self.moment_weight = params.moment_weight
        self.svm_points_num = params.svm_points_num
        self.max_batch_size_for_moment = params.max_batch_size_for_moment
        self.batch = 0
        self.batch_max = int(len(loader) / self.batch_size) + (len(loader) % self.batch_size > 0)
        self.loader = loader
        self.batchloader = iter(DataLoader(loader, batch_size=self.batch_size, shuffle=False))
        self.probs_root = os.path.join(roots.io_root, "probs")
        self.moment_root = os.path.join(roots.io_root, "moment")

        if self.epoch == 0:
            pattern = "baseline"
            ckpt_path = roots.init_ckpt
            self.probs_load_dir = os.path.join(self.probs_root, params.optim_target, pattern)
            self.moment_load_dir = os.path.join(self.moment_root, params.optim_target, pattern)
        else:
            pattern = "epoch_" + str(self.epoch) + "_alpha_" + str(self.alpha) + "_target_" + str(params.optim_target) + "_embedding_" + str(params.embedding_img_interval)
            basename = self.model_name + "_" + pattern + ".pth"
            self.probs_load_dir = os.path.join(self.probs_root, params.optim_target, pattern)
            self.moment_load_dir = os.path.join(self.moment_root, params.optim_target, pattern)
            ckpt_path = os.path.join(roots.weights_dir, basename)
        if init_net and num_classes is not None:
            if moment == False:
                self.net = load_network(self.model_name, num_classes, ckpt_path)
            else:
                print("======== you are using non-diterministic net for moment!! ========")
                self.net = load_network(self.model_name, num_classes, ckpt_path, True)

    def probs_gt_load(self, i, load_dir=None):
        if load_dir is None:
            load_dir = self.probs_load_dir
        try:
            filename = os.path.join(load_dir, "probs" + str(i) + ".hdf5")
            f_probs = h5py.File(filename, "r")
            probs = np.asarray(f_probs['probabilities'])
            gt_train = np.asarray(f_probs['gt_train_ids'])
            gt_label = np.asarray(f_probs['gt_label_ids'])
            probs = np.squeeze(probs)
            gt_train = np.squeeze(gt_train)
            gt_label = np.squeeze(gt_label)
            im_path = f_probs['image_path'][0].decode("utf8")
        except OSError:
            print("No probs file for image %d, therefore run inference..." % i)
            probs, gt_train, gt_label, im_path = self.prob_gt_calc(i)
        return probs, gt_train, gt_label, im_path

    def probs_gt_save(self, i, save_dir=None):
        if save_dir is None:
            save_dir = self.probs_load_dir
        if not os.path.exists(save_dir):
            print("Create directory:", save_dir)
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, "probs" + str(i) + ".hdf5")
        if os.path.exists(file_name):
            return
        probs, gt_train, gt_label, im_path = self.prob_gt_calc(i)
        f = h5py.File(file_name, "w")
        f.create_dataset("probabilities", data=probs)
        f.create_dataset("gt_train_ids", data=gt_train)
        f.create_dataset("gt_label_ids", data=gt_label)
        f.create_dataset("image_path", data=[im_path.encode('utf8')])
        print("file stored:", file_name)
        f.close()

    def probs_gt_load_batch(self):
        assert self.batch_size > 1, "Please use batch size > 1 or use function 'probs_gt_load()' instead, bye bye..."
        x, y, z, im_paths = next(self.batchloader)
        probs = prediction(self.net, x)
        gt_train = y.numpy()
        gt_label = z.numpy()
        self.batch += 1
        print("\rBatch %d/%d processed" % (self.batch, self.batch_max))
        sys.stdout.flush()
        return probs, gt_train, gt_label, im_paths

    def prob_gt_calc(self, i):
        x, y = self.loader[i]
        probs = np.squeeze(prediction(self.net, x.unsqueeze_(0)))
        gt_train = y.numpy()
        try:
            gt_label = np.array(Image.open(self.loader.annotations[i]).convert('L'))
        except AttributeError:
            gt_label = np.zeros(gt_train.shape)
        im_path = self.loader.images[i]
        return probs, gt_train, gt_label, im_path

    def moment_gt_calc(self, i):
        x, y = self.loader[i]
        npy = np.array(y)
        id_dist = []
        ood_dist =[]
        id_labels = list(range(19))
        if np.sum(npy == 255) >= 5e4:
            print("BEGIN CALCULATE MOMENT FOR ONE IMAGE")
            for _ in range(self.moment_num // self.max_batch_size_for_moment + 1):
                probs = np.transpose(prediction(self.net, torch.stack([x for _ in range(self.max_batch_size_for_moment)], dim=0)), (2, 3, 0, 1))
                ood_probs = np.random.permutation(probs[npy == 255])[0: int(5e4)]
                id_probs = np.random.permutation(probs[np.isin(npy, id_labels)])[0: int(5e4)]
                ood_dist.append(ood_probs)
                id_dist.append(id_probs)
            return [np.concatenate(id_dist, axis=1)], [np.concatenate(ood_dist, axis=1)]
        else:
            return id_dist, ood_dist

    def moment_test_calc(self, i):
        x, y = self.loader[i]
        npy = np.array(y)
        id_dist = []
        ood_dist =[]
        for _ in range(self.moment_num // self.max_batch_size_for_moment + 1):
            probs = np.transpose(prediction(self.net, torch.stack([x for _ in range(self.max_batch_size_for_moment)], dim=0)), (2, 3, 0, 1))
            ood_dist.append(probs[npy == 2])
            id_dist.append(probs[npy == 1])
        id_dist = np.concatenate(id_dist, axis=1)
        ood_dist = np.concatenate(ood_dist, axis=1)
        all_pixel_dist = np.concatenate([id_dist, ood_dist], axis=0)
        label = np.concatenate([np.ones(id_dist.shape[0]), np.zeros(ood_dist.shape[0])])
        high_moment = np.transpose(stats.moment(all_pixel_dist, moment=list(range(2, self.moment_oder+1)), axis=1), (1, 0, 2)).reshape([all_pixel_dist.shape[0], -1])
        mean = np.mean(all_pixel_dist, axis=1)
        entropy = np.expand_dims(np.sum(mean * np.log(mean), axis=1), axis=1)
        moment = 2 * np.concatenate([(1 - self.moment_weight)*entropy, self.moment_weight*mean, self.moment_weight*high_moment], axis=1)
        return moment, label

    def moment_gt_save(self, save_dir=None):
        if save_dir is None:
            save_dir = self.moment_load_dir
        if not os.path.exists(save_dir):
            print("Create directory:", save_dir)
            os.makedirs(save_dir)
        total_id_dist = []
        total_ood_dist = []
        for i in range(len(self.loader)):
            if(len(total_id_dist) >= self.svm_points_num // 5e4):
                break
            single_id_dist, single_ood_dist = self.moment_gt_calc(i)
            total_id_dist += single_id_dist
            total_ood_dist += single_ood_dist
        total_id_dist = np.concatenate(total_id_dist, axis=0)
        total_ood_dist = np.concatenate(total_ood_dist, axis=0)
        id_high_moment = np.transpose(stats.moment(total_id_dist, moment=list(range(2, self.moment_oder+1)), axis=1), (1, 0, 2)).reshape([total_id_dist.shape[0], -1])
        ood_high_moment = np.transpose(stats.moment(total_ood_dist, moment=list(range(2, self.moment_oder+1)), axis=1), (1, 0, 2)).reshape([total_ood_dist.shape[0], -1])
        id_mean = np.mean(total_id_dist, axis=1)
        ood_mean = np.mean(total_ood_dist, axis=1)
        id_entropy = np.expand_dims(np.sum(id_mean * np.log(id_mean), axis=1), axis=1)
        ood_entropy = np.expand_dims(np.sum(ood_mean * np.log(ood_mean), axis=1), axis=1)
        id_moment = 2 * np.concatenate([(1 - self.moment_weight)*id_entropy, self.moment_weight*id_mean, self.moment_weight*id_high_moment], axis=1)
        ood_moment = 2 * np.concatenate([(1 - self.moment_weight)*ood_entropy, self.moment_weight*ood_mean, self.moment_weight*ood_high_moment], axis=1)
        id_label = np.ones(id_moment.shape[0])
        ood_label = np.zeros(ood_moment.shape[0])
        moment_for_svm = np.concatenate([id_moment, ood_moment], axis=0)
        label_for_svm = np.concatenate([id_label, ood_label])
        file_name = os.path.join(save_dir, "moment_" + str(self.svm_points_num) + ".pkl")
        data_dict = {}
        data_dict["data"] = moment_for_svm
        data_dict["label"] = label_for_svm
        output = open(file_name, 'wb')
        pickle.dump(data_dict, output)
        print("file stored:", file_name)
        output.close()

def probs_gt_load(i, load_dir):
    try:
        filepath = os.path.join(load_dir, "probs" + str(i) + ".hdf5")
        f_probs = h5py.File(filepath, "r")
        probs = np.asarray(f_probs['probabilities'])
        gt_train = np.asarray(f_probs['gt_train_ids'])
        gt_label = np.asarray(f_probs['gt_label_ids'])
        probs = np.squeeze(probs)
        gt_train = np.squeeze(gt_train)
        gt_label = np.squeeze(gt_label)
        im_path = f_probs['image_path'][0].decode("utf8")
    except OSError:
        probs, gt_train, gt_label, im_path = None, None, None, None
        print("No probs file, see src.model_utils")
        exit()
    return probs, gt_train, gt_label, im_path
