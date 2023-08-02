import os, sys
sys.path.append("../../src")
import numpy as np
from PIL import Image
import torch
import src.utils.utils as utils
import src.data.data_transform as data_transform
import torchvision.transforms as transforms
import kornia.color.lab as lab
from PIL import Image


class RealImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, use_chroma=False, use_depth=False, use_normal=False, size=(512, 512), test=False, using_transformer=False, intrinsic=False):
        super(RealImageDataset, self).__init__()
        self.root_path = path
        if intrinsic:
            self.path = os.path.join(path, "images_intrinsic")
            self.intrinsic = True
        else:
            self.path = os.path.join(path, "images")
            self.intrinsic = False
        
        self.samples = sorted([os.path.join(self.path, x) for x in os.listdir(self.path) if x.endswith(".png")])
        
        self.ISPDataAugmentation = data_transform.ISPDataAugmentation()
        self.image_transforms = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.size = size

        self.use_chroma = use_chroma
        self.use_depth = use_depth
        self.use_normal = use_normal
        self.test = test
        
    def __getitem__(self, index):
        index = index % len(self.samples)
        sample_path = self.samples[index]

        image_path = sample_path
        if self.intrinsic:
            image_name = sample_path.split("/")[-1].split(".")[0][:-len("_albedo1")]
        else:
            image_name = sample_path.split("/")[-1].split(".")[0]
            
        material_label_path = os.path.join(self.root_path, "masks", image_name + ".exr")

        image = Image.open(image_path).convert("RGB")
        sz = self.size[0] + 1
        H, W = image.size
        if H < W:
            image = image.resize((sz, int(sz * W / H)), Image.BOX)
        else:
            image = image.resize((int(sz * H / W), sz), Image.BOX)
        image = transforms.ToTensor()(image)
        mat_label = utils.exr_to_tensor(material_label_path)        
        mat_label = transforms.Resize(self.size[0]+1, interpolation=Image.NEAREST)(mat_label)

        if image_name.startswith("t"):
            image = image[:, :, :513]
            mat_label = mat_label[:, :, :513]
        
        image, mat_label = utils.augment_data(image,  mat_label, size=self.size, flip=False, test=self.test)
        # if mat_label is all zeros, then return a random sample
        if torch.sum(mat_label) == 0:
            return self.__getitem__(index+1)
        
        rand_h, rand_w = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1])
        while mat_label[0, rand_h, rand_w] == 0 or (torch.unique(mat_label[0, rand_h-4:rand_h+4, rand_w-4:rand_w + 4]).shape[0] != 1):
            rand_h, rand_w = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1])
        
        mat_label = (mat_label == mat_label[0, rand_h, rand_w]).float()
        image = self.image_transforms(image)

        data = {"images": image,
                "mat_labels": mat_label[0, :, :],
                "reference_locations": torch.tensor([rand_h, rand_w]),
                "sample_path": sample_path}
        return data
    
    def __len__(self):
        return 10 * len(self.samples)
