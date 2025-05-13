import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter
import torch.nn as nn
import torch.nn.functional as F
from .utils import shuffle_remap
from .cyclegan_syn import cyclegan_remap
from .utils import _Affine, _NonAffine
import os
import SimpleITK as sitk





class ImageDataset(Dataset):
    def __init__(self, root,transforms_,opt,unaligned=False):
        self.transforms = transforms.Compose(transforms_)
        self.CT_files_root = sorted(glob.glob(os.path.join(root, '*', '*', 'CT_img' + '*.nii.gz')))
        self.MR_files_root = sorted(glob.glob(os.path.join(root, '*', '*', 'MR_img' + '*.nii.gz')))
        self.opt = opt
        self._Affine = _Affine
        self._NonAffine = _NonAffine
           
      
    def __getitem__(self, index):
        # Only use one modal data for evaluator training
        CT_file = self.CT_files_root[index % len(self.CT_files_root)]
        CT_arr = sitk.GetArrayFromImage(sitk.ReadImage(CT_file)).astype(np.float32)
        CT_arr = np.clip(CT_arr, -180, 400)
        CT_arr = 2 * (CT_arr - np.min(CT_arr)) / (np.max(CT_arr) - np.min(CT_arr)) - 1
        CT_arr_show = torch.from_numpy(CT_arr).unsqueeze(0)

        MR_file = self.MR_files_root[index % len(self.MR_files_root)]
        MR_arr = sitk.GetArrayFromImage(sitk.ReadImage(MR_file)).astype(np.float32)
        b = np.percentile(MR_arr, 99.7)
        t = np.percentile(MR_arr, 0.5)
        MR_arr = np.clip(MR_arr, t, b)
        MR_arr = 2 * (MR_arr - np.min(MR_arr)) / (np.max(MR_arr) - np.min(MR_arr)) - 1
        MR_arr_show = torch.from_numpy(MR_arr).unsqueeze(0)

        item_A1 = self.transforms(CT_arr)
        item_A2 = item_A1

        #### make different affine to A and B
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_A1 = self._Affine(random_numbers=random_numbers,imgs = [item_A1],padding_modes=['border'],opt = self.opt)
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_A2 = self._Affine(random_numbers=random_numbers,imgs = [item_A2],padding_modes=['border'],opt = self.opt)

        
        ############ 
        # make different non-affine to A and B
        item_A1 = self._NonAffine(imgs = [item_A1],padding_modes=['border'],opt = self.opt)
        #keep same deformation for A and B
        item_A2 = self._NonAffine(imgs = [item_A2],padding_modes=['border'],opt = self.opt)
        
        
        label = (item_A2 + 1) / 2 - (item_A1 + 1) / 2  # make (A2-A1 ) as label and keep range (-1,1)
          

        return item_A1, item_A2, CT_arr_show, label
    
    
    
    
    def __len__(self):
        return len(self.CT_files_root)
    
    

    


