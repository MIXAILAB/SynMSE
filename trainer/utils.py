import random
import time
import datetime
import sys
import yaml
from torch.autograd import Variable
import torch
from visdom import Visdom
import torch.nn.functional as F
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter
from medpy import metric
import pystrum.pynd.ndutils as nd
import torch.nn as nn
import SimpleITK as sitk

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



class Resize3D():
    def __init__(self, size_tuple, use_cv = True, mode="trilinear"):
        self.size_tuple = size_tuple
        self.use_cv = use_cv
        self.mode = mode

    def __call__(self, tensor):
        """
            Resized the tensor to the specific size

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Resized tensor
        """
        tensor = tensor.unsqueeze(0)
        if self.mode == "nearest":
            tensor = F.interpolate(tensor, size = [self.size_tuple[0],self.size_tuple[1],self.size_tuple[2]], mode=self.mode)
        else:
            tensor = F.interpolate(tensor, size = [self.size_tuple[0],self.size_tuple[1],self.size_tuple[2]], align_corners=True, mode=self.mode)

        #tensor = tensor.squeeze(0)
 
        return tensor


class ToTensor():
    def __call__(self, tensor):
        tensor = np.expand_dims(tensor, 0)
        
        return torch.from_numpy(tensor)

def tensor2image(tensor):
    image = (127.5*(tensor.cpu().float().numpy()))+127.5
    image1 = image[0]
    for i in range(1,tensor.shape[0]):
        image1 = np.hstack((image1,image[i]))
    
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    #print ('image1.shape:',image1.shape)
    return image1.astype(np.uint8)


class Logger():
    def __init__(self, env_name ,ports, n_epochs, batches_epoch, epoch):
        self.viz = Visdom(port= ports, server = "localhost", env = env_name)
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = epoch
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title': image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name],
                               opts={'title': image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
                                                                 Y=np.array([loss / self.batch]),
                                                                 opts={'xlabel': 'epochs', 'ylabel': loss_name,
                                                                       'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss / self.batch]),
                                  win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    # print ('m:',m)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
        
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = dx*dx
    dy = dy*dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d
    return d




def create_affine_transformation_matrix(n_dims, scaling=None, rotation=None, shearing=None, translation=None):
    """
        create a 4x4 affine transformation matrix from specified values
    :param n_dims: integer
    :param scaling: list of 3 scaling values
    :param rotation: list of 3 angles (degrees) for rotations around 1st, 2nd, 3rd axis
    :param shearing: list of 6 shearing values
    :param translation: list of 3 values
    :return: 4x4 numpy matrix
    """

    T_scaling = np.eye(n_dims + 1)
    T_shearing = np.eye(n_dims + 1)
    T_translation = np.eye(n_dims + 1)

    if scaling is not None:
        T_scaling[np.arange(n_dims + 1), np.arange(n_dims + 1)
                  ] = np.append(scaling, 1)

    if shearing is not None:
        shearing_index = np.ones((n_dims + 1, n_dims + 1), dtype='bool')
        shearing_index[np.eye(n_dims + 1, dtype='bool')] = False
        shearing_index[-1, :] = np.zeros((n_dims + 1))
        shearing_index[:, -1] = np.zeros((n_dims + 1))
        T_shearing[shearing_index] = shearing

    if translation is not None:
        T_translation[np.arange(n_dims), n_dims *
                      np.ones(n_dims, dtype='int')] = translation

    if n_dims == 2:
        if rotation is None:
            rotation = np.zeros(1)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        T_rot = np.eye(n_dims + 1)
        T_rot[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation[0]),
                                                                 np.sin(
                                                                     rotation[0]),
                                                                 np.sin(
                                                                     rotation[0]) * -1,
                                                                 np.cos(rotation[0])]
        return T_translation @ T_rot @ T_shearing @ T_scaling

    else:
        if rotation is None:
            rotation = np.zeros(n_dims)
        else:
            rotation = np.asarray(rotation) * (math.pi / 180)
        T_rot1 = np.eye(n_dims + 1)
        T_rot1[np.array([1, 2, 1, 2]), np.array([1, 1, 2, 2])] = [np.cos(rotation[0]),
                                                                  np.sin(
                                                                      rotation[0]),
                                                                  np.sin(
                                                                      rotation[0]) * -1,
                                                                  np.cos(rotation[0])]
        T_rot2 = np.eye(n_dims + 1)
        T_rot2[np.array([0, 2, 0, 2]), np.array([0, 0, 2, 2])] = [np.cos(rotation[1]),
                                                                  np.sin(
                                                                      rotation[1]) * -1,
                                                                  np.sin(
                                                                      rotation[1]),
                                                                  np.cos(rotation[1])]
        T_rot3 = np.eye(n_dims + 1)
        T_rot3[np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])] = [np.cos(rotation[2]),
                                                                  np.sin(
                                                                      rotation[2]),
                                                                  np.sin(
                                                                      rotation[2]) * -1,
                                                                  np.cos(rotation[2])]
        return T_translation @ T_rot3 @ T_rot2 @ T_rot1 @ T_shearing @ T_scaling

    
    
    
def smooth_loss(flow):
    dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
    dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
    dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])

    dy = dy * dy
    dx = dx * dx
    dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    grad = d / 3.0

    return grad

def Get_Ja(flow):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    return D1 - D2 + D3


def NJ_loss(ypred):
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5 * (torch.abs(Get_Ja(ypred)) - Get_Ja(ypred))
    num_neg_Jac = torch.sum(Neg_Jac > 0)
    return num_neg_Jac
    #return torch.sum(Neg_Jac)

    
    
class Transformer_3D(nn.Module):
    def __init__(self, mode):
        super(Transformer_3D, self).__init__()
        self.mode = mode

    def forward(self, src, flow):
        b = flow.shape[0]
        d = flow.shape[2]
        h = flow.shape[3]
        w = flow.shape[4]
        size = (d, h, w)
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b, 1, 1, 1, 1).cuda()
        new_locs = grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * \
                (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        warped = F.grid_sample(
            src, new_locs, align_corners=True, padding_mode = "border", mode = self.mode) #mode='nearest'

        return warped
    
    
class Transformer_3D_cpu(nn.Module):
    def __init__(self):
        super(Transformer_3D_cpu, self).__init__()

    def forward(self, src, flow,padding):
        b = flow.shape[0]
        d = flow.shape[2]
        h = flow.shape[3]
        w = flow.shape[4]
        size = (d, h, w)
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b, 1, 1, 1, 1)
        new_locs = grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * \
                (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        warped = F.grid_sample(
            src, new_locs, align_corners=True, padding_mode = padding)

        return warped    
    
def upsample_dvf(dvf, o_size):
    b = dvf.shape[0]

    d = o_size[3]
    h = o_size[4]
    w = o_size[5]

    size = (d, h, w)

    upsampled_dvf = torch.cat([F.interpolate(dvf[:, 0, :, :, :].unsqueeze(0),
                                             size,
                                             mode='trilinear', align_corners=True),

                               F.interpolate(dvf[:, 1, :, :, :].unsqueeze(0),
                                             size,
                                             mode='trilinear', align_corners=True),

                               F.interpolate(dvf[:, 2, :, :, :].unsqueeze(0),
                                             size,
                                             mode='trilinear', align_corners=True)], dim=1)
    return upsampled_dvf










from monai.transforms import RandGibbsNoise,RandGaussianNoise,RandRicianNoise,RandBiasField,RandHistogramShift,RandKSpaceSpikeNoise,RandGaussianSharpen,RandAdjustContrast#,RandIntensityRemap

augmentations = [RandGaussianSharpen(prob = 0.3),
                 RandGaussianNoise(prob =0.3),
    RandBiasField(prob = 0.3),RandAdjustContrast(prob = 0.3),RandKSpaceSpikeNoise(prob = 0.3)  
    
]

def aug_func(data):
    for _ in range(len(augmentations)):
        data = augmentations[_](data)
    return data

def histgram_shift(data):
    num_control_point = random.randint(2,8)
    reference_control_points = torch.linspace(0, 1, num_control_point)
    floating_control_points = reference_control_points.clone()
    for i in range(1, num_control_point - 1):
        floating_control_points[i] = floating_control_points[i - 1] + torch.rand(
            1) * (floating_control_points[i + 1] - floating_control_points[i - 1])
    img_min, img_max = data.min(), data.max()
    reference_control_points_scaled = (reference_control_points *
                                       (img_max - img_min) + img_min).numpy()
    floating_control_points_scaled = (floating_control_points *
                                      (img_max - img_min) + img_min).numpy()
    data_shifted = np.interp(data, reference_control_points_scaled,
                             floating_control_points_scaled)
    return data_shifted


def add_gaussian_noise(data, mean=0, std=0.3):
    image_shape = data.shape
    noise = torch.normal(mean, std, size=image_shape)
    vmin, vmax = torch.min(data), torch.max(data)
    mean, std = torch.mean(data), torch.std(data)
    data_normed = (data - mean) / std + noise
    data_normed = torch.clip(data_normed * std + mean, vmin, vmax)
    return data_normed







def shuffle_remap(data, ranges = [-1,1], rand_point = [2,50]):
    # 定义一个名为shuffle_remap的函数，它接受一个数据张量、数值范围以及控制点数量范围作为参数

    control_point = random.randint(rand_point[0], rand_point[1])
    # 生成一个控制点数量，这个数量是在rand_point[0]和rand_point[1]之间的随机整数
    distribu = torch.rand(control_point) * (ranges[1] - ranges[0]) + ranges[0]
    # 生成一个包含控制点的随机值分布，这个分布的范围是在ranges[0]和ranges[1]之间的随机值
    distribu, _ = torch.sort(distribu)
    # 对随机值分布进行排序

    ### --> -1 point1 ... pointN, 1
    distribu = torch.cat([torch.tensor([ranges[0]]), distribu])
    distribu = torch.cat([distribu, torch.tensor([ranges[1]])])
    # 将ranges[0]和ranges[1]加入到控制点值分布的两端，以确保整个范围被覆盖

    shuffle_part = torch.randperm(control_point + 1)
    # 随机生成一个长度为control_point+1的排列

    new_image = torch.zeros_like(data)
    # 创建一个和输入数据张量相同大小的全0张量，用于存储生成的新数据

    for i in range(control_point + 1):
        # 对于每个控制点，将其对应的数值范围映射到一个随机的控制点对应的范围
        # 从shuffle_part中获取第i个位置对应的随机控制点编号，表示当前控制点要映射到哪个随机控制点上。
        target_part = shuffle_part[i]
        #获取当前控制点的数值范围，即在distribu中第i和第i + 1个元素之间的值。
        min1, max1 = distribu[i], distribu[i + 1]
        # 获取当前控制点要映射到的目标控制点的数值范围，即在distribu中第target_part和第target_part+1个元素之间的值。
        min2, max2 = distribu[target_part], distribu[target_part + 1]
        # 获取所有在当前控制点数值范围内的元素的坐标。通过torch.where函数获取数据data中所有满足min1 <= data < max1条件的元素的坐标。
        coord = torch.where((min1 <= data) & (data < max1))
        # 将获取的所有坐标映射到目标数值范围内，并将结果赋值给新的数据new_image。
        # 具体地，先将这些元素通过(data[coord] - min1) / (max1 - min1)进行标准化，得到它们在当前数值范围内的相对位置，再乘以max2 - min2将其映射到目标数值范围内，最后加上min2将其调整到正确的位置。
        new_image[coord] = ((data[coord] - min1) / (max1 - min1)) * (max2 - min2) + min2

    if torch.rand(1) < 0.2:
        new_image = -new_image
    # 20%的概率将新数据张量取反
    if torch.rand(1) < 0.2:
        new_image = torch.from_numpy(histgram_shift(new_image)).to(torch.float32)
    # 20%的概率对新数据张量进行直方图偏移操作
    new_image = torch.clamp(new_image, ranges[0], ranges[1])
    # 将新数据张量的值夹紧到指定的范围内
    if torch.rand(1) < 0.2:
        new_image = torch.clamp(aug_func(new_image), 0, 1).to(torch.float32)
    # 20%的概率对新数据张量应用自定义的数据增强函数，并将结果夹紧到[0,1]范围内

    return new_image
    # 返回生成的新数据张量

def HD(image,label):
    #label_list = np.unique(image)[1:]
    label_list = [1,2,3,4]
    HD_list = []
    for i in label_list:
        haus_dic95 = metric.hd95(image == i, label == i)
        HD_list.append(haus_dic95)

    #haus_dic95 = metric.hd95(image,label)
    #return haus_dic95
    return np.mean(HD_list), HD_list

def ASSD(image,label):
    label_list = [1,2,3]
    ASSD_list = []
    for i in label_list:
        assd = metric.assd(image == i, label == i)
        ASSD_list.append(assd)

    # haus_dic95 = metric.hd95(image,label)
    # return haus_dic95
    return np.mean(ASSD_list)







def _NonAffine(imgs, padding_modes,opt,elastic_random=None):
    if elastic_random is None:
        elastic_random = torch.rand([3,imgs[0].shape[2],imgs[0].shape[3],imgs[0].shape[4]]).numpy()*2-1#.numpy()

    sigma = opt["gaussian_smoothing"]        #需要根据图像大小调整
    alpha = opt["non_affine_alpha"]  #需要根据图像大小调整
    dz = gaussian_filter(elastic_random[0], sigma) * alpha
    dx = gaussian_filter(elastic_random[1], sigma) * alpha
    dy = gaussian_filter(elastic_random[2], sigma) * alpha

    dz = np.expand_dims(dz, 0)
    dx = np.expand_dims(dx, 0)
    dy = np.expand_dims(dy, 0)

    flow = np.concatenate((dz,dx,dy), 0)
    flow = np.expand_dims(flow, 0)
    flow = torch.from_numpy(flow).to(torch.float32)

    res_img = []
    for img, mode in zip(imgs, padding_modes):
        img = Transformer_3D_cpu()(img, flow, padding = mode)
        res_img.append(img.squeeze(0))
    
    return res_img[0] if len(res_img) == 1 else res_img



def _Affine( random_numbers,imgs,padding_modes,opt):
    D, H, W = imgs[0].shape[2:]
    n_dims = 3
    tmp = np.ones(3)
    tmp[0:3] = random_numbers[0:3]
    scaling = tmp * opt['scaling'] + 1
    tmp[0:3] = random_numbers[3:6]
    rotation = tmp * opt['rotation']
    tmp[0:2] = random_numbers[6:8]
    tmp[2] = random_numbers[8]/2
    translation = tmp * opt['translation'] 
    theta = create_affine_transformation_matrix(
        n_dims=n_dims, scaling=scaling, rotation=rotation, shearing=None, translation=translation)
    theta = theta[:-1, :]
    theta = torch.from_numpy(theta).to(torch.float32)
    size = torch.Size((1, 1, D, H, W))
    grid = F.affine_grid(theta.unsqueeze(0), size, align_corners=True)

    res_img = []
    for img, mode in zip(imgs, padding_modes):
        res_img.append(F.grid_sample(img, grid, align_corners=True, padding_mode=mode))

    return res_img[0] if len(res_img) == 1 else res_img
      







