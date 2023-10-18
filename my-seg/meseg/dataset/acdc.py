import os
import random
import numpy as np
import torch
import wandb
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms
from .dataset import register_dataset
from medpy.metric import dc, hd95


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


@register_dataset
class acdc_2d(Dataset):
    classes = 4
    train_transform = transforms.Compose([
        RandomGenerator(output_size=[224,224])
    ])
    valid_test_transform = None

    def __init__(self, root="", mode="train") -> None:
        self.mode = mode
        self.data_dir = os.path.join(root, self.mode)
        self.sample_list = open(os.path.join(root, f"lists_ACDC/{self.mode}.txt")).readlines()
        if self.mode=="train":
            self.transform = self.train_transform
        else:
            self.transform = self.valid_test_transform
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        if self.mode == "train" or self.mode == "valid":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name)
            data = np.load(data_path)
            image, label = data['img'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}".format(vol_name)
            data = np.load(filepath)
            image, label = data['img'], data['label']
        
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        if self.mode=="train" or self.mode == "valid":
            return sample["image"], sample["label"]
        else:
            sample['case_name'] = self.sample_list[idx].strip('\n')
            return sample

def acdc_2d_validation(valid_dataloder, model, args):
    args.log("Validation ==========================================")
    dc_sum=0
    metric_list = 0.0
    model.eval()
    for i, val_sampled_batch in enumerate(valid_dataloder):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.type(torch.FloatTensor), val_label_batch.type(torch.FloatTensor)
        val_image_batch, val_label_batch = val_image_batch.cuda().unsqueeze(1), val_label_batch.cuda().unsqueeze(1)
        p1, p2, p3, p4 = model(val_image_batch)
        val_outputs = p1 + p2 + p3 + p4
        val_outputs = torch.argmax(torch.softmax(val_outputs, dim=1), dim=1).squeeze(0)
        
        dc_sum+=dc(val_outputs.cpu().data.numpy(),val_label_batch[:].cpu().data.numpy())
    performance = dc_sum / len(valid_dataloder)
    args.log('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, args.best))
    if args.use_wandb:
        wandb.log({"acdc valid dice score": performance})
    if performance > args.best:
        args.bet = performance
        args.log(">>>>> best model is changed <<<<<")
        torch.save(model.state_dict(), args.best_weight_path)
    args.log("="*100)