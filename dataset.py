from io import BytesIO

# import lmdb
from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as F2
import random
import torch

class MultiResolutionDataset(Dataset):
    def __init__(self, path, resolution=8):

        files = os.listdir(path)
        self.imglist =[]
        for fir in files:
            self.imglist.append(os.path.join(path,fir))

        self.transform = transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
        )

        self.resolution = resolution


    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        img = Image.open(self.imglist[index])
        img = self.transform(img)

        return img

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F2.pad(image, padding, 0, 'constant')

class IrisDataset(Dataset):
    def __init__(self, img_dir = '../images', resolution = 8):
        super(IrisDataset, self).__init__()
        self.img_dir = img_dir
        self.imgs = [x for x in os.listdir(img_dir) if x[-4:] == 'tiff']
        self.transforms = transforms.Compose([
            SquarePad(),
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], inplace=True)
        ])
        self.resolution = resolution
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        img_path = os.path.join(self.img_dir, self.imgs[i])
        img = Image.open(img_path)
        img = self.transforms(img)
        return img

class MultiLabelResolutionDataset(Dataset):
    def __init__(self, path, resolution=8):
        folders = path

        self.imglist =[]
        self.attributes = []
        for i in range(len(folders)):
            files = os.listdir(folders[i])
            for fir in files:
                self.imglist.append(os.path.join(folders[i],fir))
                self.attributes.append(i)

        c = list(zip(self.imglist,self.attributes))
        random.shuffle(c)
        self.imglist2, self.att2 = zip(*c)

        self.transform = transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
        )

        self.resolution = resolution

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        
        img = Image.open(self.imglist[index])
        img = self.transform(img)
        label_org = self.attributes[index]
        label_trg = self.att2[index]
        return img,label_org,label_trg

class MultiLabelAllDataset(Dataset):
    def __init__(self, path,spath,ppath,resolution=256):

        self.slist = []
        self.plist = []
        self.imglist = []
        self.attributes = []
        for i in range(len(spath)):
            style = os.listdir(spath[i])
            pix = os.listdir(ppath[i])
            imgs = os.listdir(path[i])
            style.sort()
            pix.sort()
            imgs.sort()
            for ii in range(len(style)):
                self.slist.append(os.path.join(spath[i],style[ii]))
                self.plist.append(os.path.join(ppath[i],pix[ii]))
                self.imglist.append(os.path.join(path[i],imgs[ii]))
                self.attributes.append(i)
            
        c = list(zip(self.imglist,self.slist,self.plist,self.attributes))
        random.shuffle(c)
        self.imglist2,self.slist2,self.plist2, self.att2 = zip(*c)
        self.transform = transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
    def __len__(self):
        return len(self.slist)

    def __getitem__(self, index):
        
        img = Image.open(self.imglist[index])
        sty = torch.load(self.slist[index])
        sty.requires_grad=False
        pix = torch.load(self.plist[index])
        pix.requires_grad=False
        
        img = self.transform(img)

        label_org = self.attributes[index]
        label_trg = self.att2[index]
        return img,sty,pix,label_org,label_trg