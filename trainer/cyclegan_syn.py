import os
import sys
from torch.utils.data import DataLoader
from Yaml.cyclegan_options import TestOptions
from model import create_model
from .cyclegan_dataset import *
import model.cyclegan_networks3D as networks3D
import torch
import math
from torch.autograd import Variable
from tqdm import tqdm
import datetime
import SimpleITK as sitk
import numpy as np

def from_numpy_to_itk(image_np, image_itk):
    image_np = np.transpose(image_np, (2, 1, 0))
    image = sitk.GetImageFromArray(image_np)
    image.SetOrigin(image_itk.GetOrigin())
    image.SetDirection(image_itk.GetDirection())
    image.SetSpacing(image_itk.GetSpacing())
    return image


def prepare_batch(image, ijk_patch_indices):
    image_batches = []
    for batch in ijk_patch_indices:
        image_batch = []
        for patch in batch:
            image_patch = image[patch[0]:patch[1], patch[2]:patch[3], patch[4]:patch[5]]
            image_batch.append(image_patch)

        image_batch = np.asarray(image_batch)
        # image_batch = image_batch[:, :, :, :, np.newaxis]
        image_batches.append(image_batch)

    return image_batches

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

def inference(model, image, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size=1):

    # create transformations to image and labels
    """
    transforms1 = [
        NiftiDataset_testing.Resample(resolution, resample)
    ]

    transforms2 = [
        NiftiDataset_testing.Padding((patch_size_x, patch_size_y, patch_size_z))
    ]
    """
    # read image file
    #image = resize_image_itk(image, newSize=(128, 96, 128), resamplemethod=sitk.sitkLinear)
    # normalize the image
    #image = Normalization(image)
    # 设置数据类型转换为float32
    image = image.astype(np.float32)
    #castImageFilter = sitk.CastImageFilter()
    #castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    #image = castImageFilter.Execute(image)

    # create empty label in pair with transformed image
    # 创建一个image同样大小的全0数组用来后续输出结果
    label_tfm = np.zeros_like(image)
    sample = {'image': image, 'label': label_tfm}


    # keeping track on how much padding will be performed before the inference
    image_array = image
    pad_x = patch_size_x - (patch_size_x - image_array.shape[2])
    pad_y = patch_size_y - (patch_size_y - image_array.shape[1])
    pad_z = patch_size_z - (patch_size_z - image_array.shape[0])

    image_pre_pad = sample['image']
    # convert image to numpy array
    image_np = sample['image']
    label_np = sample['label'].astype(np.float32)

    # unify numpy and sitk orientation
    # 用于翻转或置换数组的轴，返回修改后的数组。
    image_np = np.transpose(image_np, (2, 1, 0))
    label_np = np.transpose(label_np, (2, 1, 0))

    # ----------------- Padding the image if the z dimension still is not even ----------------------

    if (image_np.shape[2] % 2) == 0:
        Padding = False
    else:
        #np.pad的的三个参数，一个是要填充的数组，第二个是一个元组或列表，指定在每个轴上要填充的数量，第三个是指定填充方式的字符串参数。
        image_np = np.pad(image_np, ((0, 0), (0, 0), (0, 1)), 'edge')
        label_np = np.pad(label_np, ((0, 0), (0, 0), (0, 1)), 'edge')
        Padding = True

    # ------------------------------------------------------------------------------------------------

    # a weighting matrix will be used for averaging the overlapped region
    weight_np = np.zeros(label_np.shape)

    # prepare image batch indices
    inum = int(math.ceil((image_np.shape[0] - patch_size_x) / float(stride_inplane))) + 1
    jnum = int(math.ceil((image_np.shape[1] - patch_size_y) / float(stride_inplane))) + 1
    knum = int(math.ceil((image_np.shape[2] - patch_size_z) / float(stride_layer))) + 1

    patch_total = 0
    ijk_patch_indices = []
    ijk_patch_indicies_tmp = []

    for i in range(inum):
        for j in range(jnum):
            for k in range(knum):
                if patch_total % batch_size == 0:
                    ijk_patch_indicies_tmp = []

                istart = i * stride_inplane
                if istart + patch_size_x > image_np.shape[0]:  # for last patch
                    istart = image_np.shape[0] - patch_size_x
                iend = istart + patch_size_x

                jstart = j * stride_inplane
                if jstart + patch_size_y > image_np.shape[1]:  # for last patch
                    jstart = image_np.shape[1] - patch_size_y
                jend = jstart + patch_size_y

                kstart = k * stride_layer
                if kstart + patch_size_z > image_np.shape[2]:  # for last patch
                    kstart = image_np.shape[2] - patch_size_z
                kend = kstart + patch_size_z

                ijk_patch_indicies_tmp.append([istart, iend, jstart, jend, kstart, kend])

                if patch_total % batch_size == 0:
                    ijk_patch_indices.append(ijk_patch_indicies_tmp)

                patch_total += 1

    batches = prepare_batch(image_np, ijk_patch_indices)

    for i in tqdm(range(len(batches))):
        batch = batches[i]

        #batch = (batch - 127.5) / 127.5

        batch = torch.from_numpy(batch[np.newaxis, :, :, :])
        batch = Variable(batch.cuda())

        # pred = model(batch)
        pred = model(batch)
        pred = pred.squeeze().data.cpu().numpy()

        #pred = (pred * 127.5) + 127.5

        istart = ijk_patch_indices[i][0][0]
        iend = ijk_patch_indices[i][0][1]
        jstart = ijk_patch_indices[i][0][2]
        jend = ijk_patch_indices[i][0][3]
        kstart = ijk_patch_indices[i][0][4]
        kend = ijk_patch_indices[i][0][5]
        label_np[istart:iend, jstart:jend, kstart:kend] += pred[:, :, :]
        weight_np[istart:iend, jstart:jend, kstart:kend] += 1.0

    print("{}: Evaluation complete".format(datetime.datetime.now()))

    # eliminate overlapping region using the weighted value
    label_np = (np.float32(label_np) / np.float32(weight_np))

    # removed the 1 pad on z
    if Padding is True:
        label_np = label_np[:, :, 0:(label_np.shape[2] - 1)]

    # removed all the padding

    label_np = label_np[:pad_x, :pad_y, :pad_z]

    label_np = np.transpose(label_np, (2, 1, 0))
    img = sitk.GetImageFromArray(label_np)
    #sitk.WriteImage(img,"/opt/data/private/3D_CycleGan/all_checkpoints/sc_ct_mrstyle_clip/out/8.nii.gz")
    return label_np

def cyclegan_remap(image_ori):

    image = image_ori.numpy()

    netG = networks3D.define_G(input_nc = 1, output_nc = 1, ngf=64, netG = 'resnet_9blocks',
                                    norm = 'instance', use_dropout = False, init_type = 'normal', init_gain = 0.02, gpu_ids = [0])
    # Replace it with the path to your pre-trained generative model.
    load_generator_modelpath = "/opt/data/private/3D_CycleGan/clinic_checkpoints/clinic_ct_mrstyle/295_net_G_B.pth"
    state_dict = torch.load(load_generator_modelpath)
    netG.load_state_dict(state_dict)
    patch_size = [320, 192, 24]#24, 64, 64,[96, 72, 96],[64, 48, 64]
    new_image = inference(netG, image, patch_size[0], patch_size[1], patch_size[2], stride_inplane=24, stride_layer=12, batch_size=1)
    return torch.from_numpy(new_image)