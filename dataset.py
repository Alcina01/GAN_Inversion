import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import utils


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageDataset(data.Dataset):

    def __init__(self,
                 root_dir,
                 meta_file,
                 beta_file,
                 geta_file,
                 theta_file,
                 transform=None,
                 image_size=128,
                 normalize=True):
        self.root_dir = root_dir
        if transform is not None:
            self.transform = transform
        else:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            if normalize:
                self.transform = transforms.Compose([
                    utils.CenterCropLongEdge(),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std)
                ])
            else:
                self.transform = transforms.Compose([
                    utils.CenterCropLongEdge(),
                    transforms.Resize(image_size),
                    transforms.ToTensor()
                ])
            self.transform_mask = transforms.Compose([
                    utils.CenterCropLongEdge(),
                    transforms.Resize(image_size),
                    transforms.ToTensor()
                ])
        with open(meta_file) as f,open(beta_file) as b,open(geta_file) as g,open(theta_file) as t:
            meta_lines = f.readlines()
            beta_lines = b.readlines()
            geta_lines = g.readlines()
            theta_lines = t.readlines()
        print("building dataset from %s" % meta_file)
        self.num = len(meta_lines)
        self.metas = []
        self.betas = []
        self.getas = []
        self.thetas = []
        self.classifier = None
        for line in meta_lines:
            line_split = line.rstrip().split()
            if len(line_split) == 2:
                self.metas.append((line_split[0], int(line_split[1])))
            else:
                self.metas.append((line_split[0], -1))
        for line in beta_lines:
            line_split = line.rstrip().split()
            if len(line_split) == 2:
                self.betas.append((line_split[0], int(line_split[1])))
            else:
                self.betas.append((line_split[0], -1))
        for line in geta_lines:
            line_split = line.rstrip().split()
            if len(line_split) == 2:
                self.getas.append((line_split[0], int(line_split[1])))
            else:
                self.getas.append((line_split[0], -1))
        for line in theta_lines:
            line_split = line.rstrip().split()
            if len(line_split) == 2:
                self.thetas.append((line_split[0], int(line_split[1])))
            else:
                self.thetas.append((line_split[0], -1))
        print("read meta done")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = self.root_dir + '/' + self.metas[idx][0]
        cls = self.metas[idx][1]
        img = default_loader(filename)
        
        mask_filename = self.root_dir + '/' + self.betas[idx][0]
        mask = default_loader(mask_filename)

        gt_filename = self.root_dir + '/' + self.getas[idx][0]
        gt = default_loader(gt_filename)
        

        mask_gt_filename = self.root_dir + '/' + self.thetas[idx][0]
        maskpair = default_loader(mask_gt_filename)

        # transform
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform_mask(mask)
            maskpair = self.transform_mask(maskpair)
            gt = self.transform(gt)
            mask[mask>0] = 1
            maskpair[maskpair>0] = 1
        mask = 1-mask
        # import ipdb;ipdb.set_trace()
        return img, cls, self.metas[idx][0],mask , self.betas[idx][0],gt,self.getas[idx][0],maskpair
