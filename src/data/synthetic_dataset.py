import os, sys
sys.path.append("../../src")
import numpy as np
import torch
import src.utils.utils as utils
import src.data.data_transform as data_transform
import torchvision.transforms as transforms
import kornia.color.lab as lab


class SyntheticMaterialDataset(torch.utils.data.Dataset):
    def __init__(self, path, stage="train", use_chroma=False, use_depth=False, use_normal=False, size=(512, 512), test=False):
        super(SyntheticMaterialDataset, self).__init__()
        self.path = os.path.join(path, stage)
        self.samples = sorted([os.path.join(self.path, x) for x in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, x))])
        self.ISPDataAugmentation = data_transform.ISPDataAugmentation()
        self.image_transforms = transforms.Compose([transforms.ColorJitter(contrast=0.1, saturation=0.1),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   
        self.size = size
        self.test = test
        
    def __getitem__(self, index):
        sample_path = self.samples[index]
        files = [os.path.join(sample_path, x) for x in os.listdir(sample_path)]
        if len(files) < 2:
            return self.__getitem__(np.random.randint(0, len(self.samples)))
        image_path = utils.get_path_with_name(files, "Image")
        material_label_path = utils.get_path_with_name(files, "segmentation")

        try:
            image = utils.exr_to_tensor(image_path)
            image = utils.correct_exposure(image)
            image = self.ISPDataAugmentation(image)
            mat_label = utils.exr_to_tensor(material_label_path)
                
            image,mat_label = utils.augment_data(image, mat_label, size=self.size, test=self.test)
            image = self.image_transforms(image)
            
            if len(torch.unique(mat_label)) < 2:
                return self.__getitem__(np.random.randint(0, len(self.samples)))
            
            rand_h, rand_w = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1])
            while mat_label[0, rand_h, rand_w] == 0:
                rand_h, rand_w = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1])
            
            mat_label = (mat_label == mat_label[0, rand_h, rand_w]).float()
            # dictionary with image, material_label, reference_locations
            data = {"images": image,
                    "mat_labels": mat_label[0, :, :],
                    "reference_locations": torch.tensor([rand_h, rand_w])}

            for key in data:
                # detect nan, if nan, print the key and return a different sample
                if torch.isnan(data[key]).any():
                    print("Detected nan", index, self.samples[index], "key:", key)
                    print("data[key]:", data[key])
                    return self.__getitem__(np.random.randint(0, len(self.samples)))
        except:
            return self.__getitem__(np.random.randint(0, len(self.samples)))
        return data
    
    def __len__(self):
        return len(self.samples)