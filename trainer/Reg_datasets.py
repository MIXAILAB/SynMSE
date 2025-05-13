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
from .utils import _Affine, _NonAffine
import os
import SimpleITK as sitk

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkLinear):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  #获取原图size
    originSpacing = itkimage.GetSpacing()  #获取原图spacing
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)   #spacing格式转换
    resampler.SetReferenceImage(itkimage)   #指定需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  #得到重新采样后的图像
    return itkimgResampled

class ImageDataset(Dataset):
    def __init__(self, root,transforms_,opt,unaligned=False):
        self.transforms = transforms.Compose(transforms_)
        #self.files_root = sorted(glob.glob("%s/*" % root))
        #self.files_root = sorted([os.path.join(root, x) for x in os.listdir(root)])
        self.CT_files_root = sorted(glob.glob(os.path.join(root, '*', '*', 'CT_img' + '*.nii.gz')))
        self.MR_files_root = sorted(glob.glob(os.path.join(root, '*', '*', 'MR_img' + '*.nii.gz')))
        self.opt = opt
        self._Affine = _Affine
        self._NonAffine = _NonAffine
    def __getitem__(self, index):

        CT_file = self.CT_files_root[index % len(self.CT_files_root)]
        MR_file = self.MR_files_root[index % len(self.MR_files_root)]

        CT_arr = sitk.GetArrayFromImage(sitk.ReadImage(CT_file)).astype(np.float32)
        MR_arr = sitk.GetArrayFromImage(sitk.ReadImage(MR_file)).astype(np.float32)

        CT_arr = np.clip(CT_arr,-180,400)
        b = np.percentile(MR_arr, 99.7)
        t = 60.0
        MR_arr = np.clip(MR_arr, t, b)

        CT_arr = 2 * (CT_arr-np.min(CT_arr))/(np.max(CT_arr)-np.min(CT_arr)) - 1
        MR_arr = 2 * (MR_arr - np.min(MR_arr)) / (np.max(MR_arr) - np.min(MR_arr)) - 1


        item_A = self.transforms(CT_arr)
        item_B = self.transforms(MR_arr)
        #### make different affine to A and B
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_A = self._Affine(random_numbers=random_numbers,imgs = [item_A],padding_modes=['border'],opt = self.opt)
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_B = self._Affine(random_numbers=random_numbers,imgs = [item_B],padding_modes=['border'],opt = self.opt)
        ############ 
        # make different non-affine to A and B
        item_A = self._NonAffine(imgs = [item_A],padding_modes=['border'],opt = self.opt)
        #keep same deformation for A and B
        item_B = self._NonAffine(imgs = [item_B],padding_modes=['border'],opt = self.opt)
        return item_A, item_B
    def __len__(self):
        return len(self.MR_files_root)

    
    
    
    




##############test################
class TestDataset(Dataset):
    def __init__(self, root,transforms_,opt,unaligned=False):
        self.transforms = transforms.Compose(transforms_)
        self.CT_files_root = sorted(glob.glob(os.path.join(root, '*_0001.nii.gz')))
        self.MR_files_root = sorted(glob.glob(os.path.join(root, '*_0000.nii.gz')))
        self.CT_masks_root = sorted(glob.glob(os.path.join(root, 'label', '*_0001.nii.gz')))
        self.MR_masks_root = sorted(glob.glob(os.path.join(root, 'label', '*_0000.nii.gz')))
        self.opt = opt
        self._Affine = _Affine
        self._NonAffine = _NonAffine
    
    def __getitem__(self, index):
        CT_file = self.CT_files_root[index % len(self.CT_files_root)]
        MR_file = self.MR_files_root[index % len(self.MR_files_root)]
        CT_mask = self.CT_masks_root[index % len(self.CT_masks_root)]
        MR_mask = self.MR_masks_root[index % len(self.MR_masks_root)]

        CT_arr = sitk.GetArrayFromImage(sitk.ReadImage(CT_file)).astype(np.float32)
        MR_arr = sitk.GetArrayFromImage(sitk.ReadImage(MR_file)).astype(np.float32)
        CT_mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(CT_mask)).astype(np.float32)
        MR_mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(MR_mask)).astype(np.float32)
        '''
        CT_arr = np.clip(CT_arr, -180, 400)
        b = np.percentile(MR_arr, 95)
        t = np.percentile(MR_arr, 0)
        MR_arr = np.clip(MR_arr, t, b)
        '''
        CT_arr = 2 * (CT_arr - np.min(CT_arr)) / (np.max(CT_arr) - np.min(CT_arr)) - 1
        MR_arr = 2 * (MR_arr - np.min(MR_arr)) / (np.max(MR_arr) - np.min(MR_arr)) - 1

        item_A = self.transforms(CT_arr)
        item_B = self.transforms(MR_arr)
        
        # Only use one modal data for evaluator training

        item_Mask_A = self.transforms(MR_mask_arr) # MASK
        item_Mask_B = self.transforms(CT_mask_arr) # MASK
        #item_Mask_A[item_Mask_A > 0] = 1
        #item_Mask_B[item_Mask_B > 0] = 1
        """
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_A ,item_Mask_A = self._Affine(random_numbers=random_numbers,imgs = [item_A, item_Mask_A],
                                            padding_modes=['border', 'zeros'],opt=self.opt)
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_B, item_Mask_B = self._Affine(random_numbers=random_numbers,imgs = [item_B, item_Mask_B],
                                            padding_modes=['border', 'zeros'],opt=self.opt)
        
#       ############
#       # deformation
        item_A, item_Mask_A = self._NonAffine(imgs = [item_A,item_Mask_A],padding_modes=['border', 'zeros'],opt=self.opt)
        item_B, item_Mask_B = self._NonAffine(imgs = [item_B,item_Mask_B],padding_modes=['border', 'zeros'],opt=self.opt)
        """
        item_A = item_A[:, 0]
        item_B = item_B[:, 0]
        item_Mask_A = item_Mask_A[:, 0]
        item_Mask_B = item_Mask_B[:, 0]
        return item_A, item_B, item_Mask_A, item_Mask_B
    
    
    
    
    def __len__(self):
        return len(self.CT_files_root)


class ImageDataset_brats(Dataset):
    def __init__(self, root, transforms_, opt, unaligned=False):
        self.transforms = transforms.Compose(transforms_)
        # self.files_root = sorted(glob.glob("%s/*" % root))
        # self.files_root = sorted([os.path.join(root, x) for x in os.listdir(root)])
        self.T1_files_root = sorted(glob.glob(os.path.join(root, 'BraTS19*', '*_t1.nii.gz')))
        self.T2_files_root = sorted(glob.glob(os.path.join(root, 'BraTS19*', '*_t2.nii.gz')))
        self.opt = opt
        self._Affine = _Affine
        self._NonAffine = _NonAffine

    def __getitem__(self, index):
        T1_file = self.T1_files_root[index % len(self.T1_files_root)]
        T2_file = self.T1_files_root[index % len(self.T2_files_root)]

        T1_arr = sitk.GetArrayFromImage(sitk.ReadImage(T1_file)).astype(np.float32)
        T2_arr = sitk.GetArrayFromImage(sitk.ReadImage(T2_file)).astype(np.float32)

        b1 = np.percentile(T1_arr, 99)
        t1 = np.percentile(T1_arr, 0)
        T1_arr = np.clip(T1_arr, t1, b1)

        b2 = np.percentile(T2_arr, 99)
        t2 = np.percentile(T2_arr, 0)
        T2_arr = np.clip(T2_arr, t2, b2)

        T1_arr = 2 * (T1_arr - np.min(T1_arr)) / (np.max(T1_arr) - np.min(T1_arr)) - 1
        T2_arr = 2 * (T2_arr - np.min(T2_arr)) / (np.max(T2_arr) - np.min(T2_arr)) - 1

        item_A = self.transforms(T1_arr)
        item_B = self.transforms(T2_arr)
        #### make different affine to A and B
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_A = self._Affine(random_numbers=random_numbers, imgs=[item_A], padding_modes=['border'], opt=self.opt)
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_B = self._Affine(random_numbers=random_numbers, imgs=[item_B], padding_modes=['border'], opt=self.opt)
        ############
        # make different non-affine to A and B
        item_A = self._NonAffine(imgs=[item_A], padding_modes=['border'], opt=self.opt)
        # keep same deformation for A and B
        item_B = self._NonAffine(imgs=[item_B], padding_modes=['border'], opt=self.opt)
        return item_A, item_B

    def __len__(self):
        return len(self.T1_files_root)


##############test################
class TestDataset_brats(Dataset):
    def __init__(self, root, transforms_, opt, unaligned=False):
        self.transforms = transforms.Compose(transforms_)
        self.T1_files_root = sorted(glob.glob(os.path.join(root, 'BraTS19*', '*_t1.nii.gz')))
        self.T2_files_root = sorted(glob.glob(os.path.join(root, 'BraTS19*', '*_t2.nii.gz')))
        self.mask_files_root = sorted(glob.glob(os.path.join(root, 'BraTS19*', '*_seg.nii.gz')))
        self.opt = opt
        self._Affine = _Affine
        self._NonAffine = _NonAffine

    def __getitem__(self, index):
        T1_file = self.T1_files_root[index % len(self.T1_files_root)]
        T2_file = self.T1_files_root[index % len(self.T2_files_root)]

        T1_arr = sitk.GetArrayFromImage(sitk.ReadImage(T1_file)).astype(np.float32)
        T2_arr = sitk.GetArrayFromImage(sitk.ReadImage(T2_file)).astype(np.float32)

        b1 = np.percentile(T1_arr, 99)
        t1 = np.percentile(T1_arr, 0)
        T1_arr = np.clip(T1_arr, t1, b1)

        b2 = np.percentile(T2_arr, 99)
        t2 = np.percentile(T2_arr, 0)
        T2_arr = np.clip(T2_arr, t2, b2)

        T1_arr = 2 * (T1_arr - np.min(T1_arr)) / (np.max(T1_arr) - np.min(T1_arr)) - 1
        T2_arr = 2 * (T2_arr - np.min(T2_arr)) / (np.max(T2_arr) - np.min(T2_arr)) - 1

        item_A = self.transforms(T1_arr)
        item_B = self.transforms(T2_arr)

        mask = self.mask_files_root[index % len(self.mask_files_root)]
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask)).astype(np.float32)
        item_Mask_A = self.transforms(mask_arr)  # MASK
        item_Mask_B = self.transforms(mask_arr)  # MASK
        item_Mask_A[item_Mask_A > 0] = 1
        item_Mask_B[item_Mask_B > 0] = 1

        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_A, item_Mask_A = self._Affine(random_numbers=random_numbers, imgs=[item_A, item_Mask_A],
                                           padding_modes=['border', 'zeros'], opt=self.opt)
        random_numbers = torch.rand(9).numpy() * 2 - 1
        item_B, item_Mask_B = self._Affine(random_numbers=random_numbers, imgs=[item_B, item_Mask_B],
                                           padding_modes=['border', 'zeros'], opt=self.opt)

        #         ############
        #         # deformation
        item_A, item_Mask_A = self._NonAffine(imgs=[item_A, item_Mask_A], padding_modes=['border', 'zeros'],
                                              opt=self.opt)
        item_B, item_Mask_B = self._NonAffine(imgs=[item_B, item_Mask_B], padding_modes=['border', 'zeros'],
                                              opt=self.opt)

        return item_A, item_B, item_Mask_A, item_Mask_B

    def __len__(self):
        return len(self.T1_files_root)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
