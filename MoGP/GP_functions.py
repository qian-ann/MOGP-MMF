

import torch
import torch.nn.functional as F
import math



class Fe1:
    def __init__(ndarray):
        pass

class Fe2:
    def __init__(ndarray):
        pass

class OFe1:
    def __init__(ndarray):
        pass

class Int1:
    def __init__(int):
        pass


class Int2:
    def __init__(int):
        pass

class Int3:
    def __init__(int):
        pass

class Float1:
    def __init__(int):
        pass

def root_con(*args):
    feature_vector = torch.concatenate((args), axis=-1)
    feature_vector=F.normalize(feature_vector, p=2, dim=-1, eps=1e-5)  #L2归一化
    return feature_vector


def root_con_maxP(*args):
    with torch.no_grad():
        feature_vector = torch.concatenate((args), axis=-1)
        num0=len(args)
        len0=len(feature_vector.shape)
        while len(feature_vector.shape) < 3:
            # 为了对最后一维（features 维度）进行池化，我们需要将 x 扩展为三维
            feature_vector = feature_vector.unsqueeze(1)  # 变为 (batch_size, 1, features)
        # 对最后一维进行最大池化（kernel_size=2, stride=2 可以调整）
        feature_vector = F.max_pool1d(feature_vector, kernel_size=num0, stride=num0)
        while len(feature_vector.shape) > len0:
            # 如果需要，可以将结果恢复为二维
            feature_vector = feature_vector.squeeze(1)
        feature_vector=F.normalize(feature_vector, p=2, dim=-1, eps=1e-5)  #L2归一化
    return feature_vector


def mixconadd(fe1, w1, fe2, w2):
    with torch.no_grad():
        # w1=w1/1000
        # w2=w2/1000
        fe=torch.add(fe1*w1,fe2*w2)
        fe=F.normalize(fe, p=2, dim=-1, eps=1e-5)  #L2归一化
    return fe

def mixconsub(fe1, w1, fe2, w2):
    with torch.no_grad():
        # w1=w1/1000
        # w2=w2/1000
        fe=torch.subtract(fe1*w1,fe2*w2)
        fe=F.normalize(fe, p=2, dim=-1, eps=1e-5)  #L2归一化
    return fe

def mul(fe1, fe2):
    with torch.no_grad():
        fe=fe1*fe2
        fe=F.normalize(fe, p=2, dim=-1, eps=1e-5)  #L2归一化
    return fe

def grt(fe1, fe2):
    with torch.no_grad():
        fe=torch.max(fe1,fe2)
        fe=F.normalize(fe, p=2, dim=-1, eps=1e-5)  #L2归一化
    return fe

def let(fe1, fe2):
    with torch.no_grad():
        fe=torch.min(fe1,fe2)
        fe=F.normalize(fe, p=2, dim=-1, eps=1e-5)  #L2归一化
    return fe

def sqrt(x):
    with torch.no_grad():
        # 创建一个张量，分别处理正数和负数
        fe = torch.where(x >= 0, torch.sqrt(x), -torch.sqrt(torch.abs(x)))
        fe=F.normalize(fe, p=2, dim=-1, eps=1e-5)  #L2归一化
    return fe

def log(x):
    with torch.no_grad():
        # 创建一个张量，分别处理正数和负数
        ero=torch.tensor(1e-4).to(x.device)
        fe = torch.where(x >= 0, torch.log(x+ero), -torch.log(torch.abs(x)+ero))
        fe=F.normalize(fe, p=2, dim=-1, eps=1e-5)  #L2归一化
    return fe

def exp(x):
    with torch.no_grad():
        # 创建一个张量，分别处理正数和负数
        fe = torch.exp(x)
        fe=F.normalize(fe, p=2, dim=-1, eps=1e-5)  #L2归一化
    return fe

def sin(x):
    with torch.no_grad():
        # 创建一个张量，分别处理正数和负数
        fe = torch.sin(x)
        fe=F.normalize(fe, p=2, dim=-1, eps=1e-5)  #L2归一化
    return fe

def cos(x):
    with torch.no_grad():
        # 创建一个张量，分别处理正数和负数
        fe = torch.cos(x)
        fe=F.normalize(fe, p=2, dim=-1, eps=1e-5)  #L2归一化
    return fe

def relu(left):
    with torch.no_grad():
        left=torch.relu(left)
        left=F.normalize(left, p=2, dim=-1, eps=1e-5)  #L2归一化
    return left


def median_filter2d(x, kernel_size):
    with torch.no_grad():
        # Padding 为了保持输入和输出大小相同
        pad_size = kernel_size // 2
        x_padded = F.pad(x.unsqueeze(0), (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0).squeeze(0)
        # 使用 unfold 提取每个窗口的局部区域
        unfolded = F.unfold(x_padded.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size)
        # 对每个局部区域进行中值操作
        median_vals = torch.median(unfolded, dim=1)[0]
        # 折叠回原始大小
        filtered = F.fold(median_vals.unsqueeze(0), output_size=x.shape, kernel_size=1)
        filtered=F.normalize(filtered, p=2, dim=-1, eps=1e-5)  #L2归一化
    return filtered.squeeze(0).squeeze(0)

def mean_filter2d(x, kernel_size):
    with torch.no_grad():
        # Padding 为了保持输入和输出大小相同
        pad_size = kernel_size // 2
        x_padded = F.pad(x.unsqueeze(0), (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0).squeeze(0)
        # 使用 unfold 提取每个窗口的局部区域
        unfolded = F.unfold(x_padded.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size)
        # 对每个局部区域进行中值操作
        median_vals = torch.mean(unfolded, dim=1)[0]
        # 折叠回原始大小
        filtered = F.fold(median_vals.unsqueeze(0), output_size=x.shape, kernel_size=1)
        filtered=F.normalize(filtered, p=2, dim=-1, eps=1e-5)  #L2归一化
    return filtered.squeeze(0).squeeze(0)

def max_filter2d(x, kernel_size):
    with torch.no_grad():
        # Padding 为了保持输入和输出大小相同
        pad_size = kernel_size // 2
        x_padded = F.pad(x.unsqueeze(0), (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0).squeeze(0)
        # 使用 unfold 提取每个窗口的局部区域
        unfolded = F.unfold(x_padded.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size)
        # 对每个局部区域进行中值操作
        median_vals = torch.max(unfolded, dim=1)[0]
        # 折叠回原始大小
        filtered = F.fold(median_vals.unsqueeze(0), output_size=x.shape, kernel_size=1)
        filtered=F.normalize(filtered, p=2, dim=-1, eps=1e-5)  #L2归一化
    return filtered.squeeze(0).squeeze(0)


def min_filter2d(x, kernel_size):
    with torch.no_grad():
        # Padding 为了保持输入和输出大小相同
        pad_size = kernel_size // 2
        x_padded = F.pad(x.unsqueeze(0), (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0).squeeze(0)
        # 使用 unfold 提取每个窗口的局部区域
        unfolded = F.unfold(x_padded.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size)
        # 对每个局部区域进行中值操作
        median_vals = torch.min(unfolded, dim=1)[0]
        # 折叠回原始大小
        filtered = F.fold(median_vals.unsqueeze(0), output_size=x.shape, kernel_size=1)
        filtered=F.normalize(filtered, p=2, dim=-1, eps=1e-5)  #L2归一化
    return filtered.squeeze(0).squeeze(0)


def gaussian_kernel1d(kernel_size, sigma):
    # 保证 kernel_size 为奇数
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    # 生成坐标轴，中心点为0
    x = torch.arange(kernel_size) - kernel_size // 2
    # 计算一维高斯核
    gauss_kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    # 归一化
    gauss_kernel = gauss_kernel / gauss_kernel.sum()
    return gauss_kernel

def gaussian_kernel2d(kernel_size, sigma):
    # 生成1D高斯核
    gauss_kernel1d = gaussian_kernel1d(kernel_size, sigma)
    # 使用1D高斯核生成2D高斯核
    gauss_kernel2d = gauss_kernel1d[:, None] @ gauss_kernel1d[None, :]
    return gauss_kernel2d

def gau(img, sigma):
    # 根据 sigma 自动确定 kernel_size (通常为 6*sigma 取最近的奇数)
    kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    # 生成2D高斯核
    gauss_kernel2d = gaussian_kernel2d(kernel_size, sigma).to(img.device)
    # 将2D高斯核调整为适合卷积的形式（C_out, C_in, H, W）
    gauss_kernel2d = gauss_kernel2d.expand(1, 1, kernel_size, kernel_size)
    # 对输入图像进行填充（为了保持尺寸不变）
    img_padded = F.pad(img, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode='constant', value=0)
    # 进行卷积操作，应用高斯滤波
    blurred_img = F.conv2d(img_padded.unsqueeze(0).unsqueeze(0), gauss_kernel2d)
    blurred_img = F.normalize(blurred_img, p=2, dim=-1, eps=1e-5)  # L2归一化
    return blurred_img.squeeze()

def laplace(img, variant="8"):
    """
    对输入图像应用拉普拉斯滤波。
    参数:
    img (Tensor): 输入的2D图像张量，形状为 (H, W)。
    variant (str): 使用哪种拉普拉斯核 ("4" 或 "8")，对应两种不同的拉普拉斯滤波器。
    返回:
    Tensor: 滤波后的图像张量。
    """
    if variant == "4":
        # 4邻域的拉普拉斯核
        kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32)
    elif variant == "8":
        # 8邻域的拉普拉斯核
        kernel = torch.tensor([[1, 1, 1],
                               [1, -8, 1],
                               [1, 1, 1]], dtype=torch.float32)
    else:
        raise ValueError("variant must be '4' or '8'")

    # 将kernel调整为适合卷积操作的形状 (C_out, C_in, H, W)
    kernel = kernel.expand(1, 1, 3, 3).to(img.device)
    # 对输入图像进行填充操作，确保输出大小与输入相同
    img_padded = F.pad(img.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant', value=0)
    # 应用卷积操作
    laplace_img = F.conv2d(img_padded, kernel)
    laplace_img = F.normalize(laplace_img, p=2, dim=-1, eps=1e-5)  # L2归一化
    # 移除多余的维度
    return laplace_img.squeeze()


# Gaussian-Laplace 过滤器实现
def gaussian_laplace(img, sigma, variant="8"):
    """应用Gaussian-Laplace滤波"""
    # 根据 sigma 自动确定 kernel_size (通常为 6*sigma 取最近的奇数)
    kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    # 生成高斯核和拉普拉斯核
    gauss_kernel2d = gaussian_kernel2d(kernel_size, sigma).to(img.device)
    if variant == "4":
        # 4邻域的拉普拉斯核
        kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32)
    elif variant == "8":
        # 8邻域的拉普拉斯核
        kernel = torch.tensor([[1, 1, 1],
                               [1, -8, 1],
                               [1, 1, 1]], dtype=torch.float32)
    else:
        raise ValueError("variant must be '4' or '8'")
    laplace_kernel2d = kernel.to(img.device)
    # 将高斯核调整为卷积形式（C_out, C_in, H, W）
    gauss_kernel2d = gauss_kernel2d.expand(1, 1, kernel_size, kernel_size)
    laplace_kernel2d = laplace_kernel2d.expand(1, 1, 3, 3)
    # 对输入图像进行高斯平滑（卷积）
    img_padded = F.pad(img.unsqueeze(0).unsqueeze(0),
                       (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
                       mode='constant', value=0)
    smoothed_img = F.conv2d(img_padded, gauss_kernel2d)
    smoothed_img = F.pad(smoothed_img,  (1, 1, 1, 1), mode='constant', value=0)
    # 然后对平滑后的图像应用拉普拉斯卷积
    laplace_img = F.conv2d(smoothed_img, laplace_kernel2d)
    laplace_img = F.normalize(laplace_img, p=2, dim=-1, eps=1e-5)  # L2归一化
    return laplace_img.squeeze()

def fft1(fe,mode=1):
    fft_fe = torch.fft.fft(fe,dim=0)
    if mode==1:
        fft_fe = torch.abs(fft_fe)  #频域振幅
    else:
        fft_fe = torch.angle(fft_fe)  # 频域相位
    fft_fe = F.normalize(fft_fe, p=2, dim=-1, eps=1e-5)  # L2归一化
    return fft_fe

def fft2(fe,mode=1):
    fft_fe = torch.fft.fft2(fe,dim=0)
    if mode==1:
        fft_fe = torch.abs(fft_fe)  #频域振幅
    else:
        fft_fe = torch.angle(fft_fe)  # 频域相位
    fft_fe = F.normalize(fft_fe, p=2, dim=-1, eps=1e-5)  # L2归一化
    return fft_fe


def eye(fe):
    return fe











