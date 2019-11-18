import time
import cv2 as cv
import torch
import numpy as np
import random
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from numpy.linalg import inv
from config import batch_size, num_workers
from data_gen import DeepHNDataset
from mobilenet_v2 import MobileNetV2
from utils import AverageMeter

device = torch.device('cpu')

def data_maker(file_full_path):
    size = (640, 480)
    patch_size = 320
    rho = 64

    Image = cv.imread(file_full_path, 0)
    Image = cv.resize(Image, size)

    ImageN = cv.imread(file_full_path, 1)
    ImageN = perturbed_image(ImageN)
    ImageN = cv.cvtColor(ImageN, cv.COLOR_BGR2GRAY)
    ImageN = cv.resize(ImageN, size)
  
    top_point = (rho, rho)
    left_point = (top_point[0], patch_size + top_point[1])
    bottom_point = (patch_size + top_point[0], patch_size + top_point[1])
    right_point = (patch_size + top_point[0], top_point[1])

    four_points = [top_point, left_point, bottom_point, right_point]

    perturbed_four_points = []
    flag = 0
    for point in four_points:
        if (flag == 0):
            perturbed_four_points.append((point[0] + random.randint(-rho, 0), point[1] + random.randint(-rho, 0)))
        elif(flag == 1):
            perturbed_four_points.append((point[0] + random.randint(-rho, 0), point[1] + random.randint(0, +rho)))
        elif(flag == 2):
            perturbed_four_points.append((point[0] + random.randint(0, +rho), point[1] + random.randint(0, +rho)))
        elif(flag == 3):
            perturbed_four_points.append((point[0] + random.randint(0, +rho), point[1] + random.randint(-rho, 0)))
        flag = flag + 1
    
    H = cv.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = inv(H)

    warped_image = cv.warpPerspective(ImageN, H_inverse, size)

    Ip1 = Image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
    Ip2 = warped_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]

    return Ip1, Ip2

def contrast(image):
    alpha = random.uniform(0.8, 1.2)

    if alpha< 1:
        beta = random.randint(-20, 0)
    if alpha > 1:
        beta = random.randint(0, 20)
    
    rows, cols, channels = image.shape

    blank = np.zeros([rows, cols, channels], image.dtype)
    dst = cv.addWeighted(image, alpha, blank, 1-alpha, beta )
    return dst

def Hsv(image):
    hue_vari = 1
    sat_vari = 0.5
    val_vari = 0.5
    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)

    img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    
    dst = cv.cvtColor(np.round(img_hsv).astype(np.uint8), cv.COLOR_HSV2BGR)
    return dst

def Gamma(image):
    gamma_vari = 0.1
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    dst = cv.LUT(image, gamma_table)
    return dst


def Motion_blur(image):
    image = np.array(image)
    degree_ = 25
    angle_ = 45
    degree = int(np.random.uniform(1, degree_))
    angle = int(np.random.uniform(-angle_, angle_))
    M = cv.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv.filter2D(image, -1, motion_blur_kernel)
    cv.normalize(blurred, blurred, 0, 255, cv.NORM_MINMAX)
    dst = np.array(blurred, dtype=np.uint8)
    return dst

def Gaussian_blur(image):

    kernel = [random.randint(1, 5) * 2 + 1 for x in range(1)]
    dst = cv.GaussianBlur(image, ksize=(kernel[0], kernel[0]), sigmaX=0, sigmaY=0)
    return dst

def perturbed_image(image):
    bet1 = np.random.uniform(1, 10)
    if bet1%2 == 0:
        image = contrast(image)
    else:
        image = Hsv(image)
    bet2 = np.random.uniform(1,10)
    if bet2%2 == 0:
        image = Motion_blur(image)
    else:
        image = Gaussian_blur(image)
    return image




if __name__ == '__main__':
    
    filename = 'model/191115_homonet.pt'
    model = MobileNetV2()
    elapsed = 0
    start = time.time()
    model.load_state_dict(torch.load(filename))
    end = time.time()
    elapsed = elapsed + (end - start)
    print('Load model {0:.5f}'.format(elapsed * 1000))
    model.eval()

    imsize = 128 if torch.cuda.is_available() else 128  # use small size if no gpu

    
    img0, img1 = data_maker("C:/Users/xwen2/Desktop/HomographyNet_supervised/inference/1.jpg")
    
    #img0 = cv.imread("C:/Users/xwen2/Desktop/HomographyNet_supervised/inference/src.jpg", 0)
    #img1 = cv.imread("C:/Users/xwen2/Desktop/HomographyNet_supervised/inference/dst.jpg", 0)

    cv.imshow("a", img0)
    cv.imshow("b", img1)
    cv.waitKey()

    img0 = cv.resize(img0, (128, 128))
    img1 = cv.resize(img1, (128, 128))

    img = np.zeros((128, 128, 3), np.float32)
    img[:, :, 0] = img0 / 255.
    img[:, :, 1] = img1 / 255.
    #img = np.transpose(img, (1, 0, 2))  # HxWxC array to CxHxW

    transformation = transforms.Compose([transforms.ToTensor()])   # covert np.array into tensor
    img_Tensor = transformation(img)
    img_Tensor=torch.unsqueeze(img_Tensor, 0)  # append extra dimention on the tensor

    img_Tensor = img_Tensor.to(device)  # [N, 3, 128, 128]
    
    
    elapsed = 0
    with torch.no_grad():
        start = time.time()
        out = model(img_Tensor)  # [N, 8]
        end = time.time()
        elapsed = elapsed + (end - start)
    print('Inference: {0:.5f} ms'.format(elapsed * 1000))

    num_out = out.numpy()

    src = np.array([
        [0,     0],
        [0,     0+128],
        [0+128, 0+128],
        [0+128, 0]], dtype="float32")

    diff = np.array([
		[num_out[0][0], num_out[0][1]],
		[num_out[0][2], num_out[0][3]],
		[num_out[0][4], num_out[0][5]],
		[num_out[0][6], num_out[0][7]]], dtype = "float32")

    estimated_perturbed = np.add(src, diff)

    M = cv.getPerspectiveTransform(src, estimated_perturbed)

    #print("Estimated displacement: \n", diff)
    #print("Perspective Matrix: \n", M)
    print('Elapsed: {0:.5f} ms'.format(elapsed * 1000))

    warped = cv.warpPerspective(img1, (M), (128,128))

    
    #img0 = cv.resize(img0, (256, 256))
    #img1 = cv.resize(img1, (256, 256))
    #warped = cv.resize(warped, (256, 256))

    img = np.zeros((128, 128, 3), np.float32)
    img[:, :, 0] = img0/255.
    img[:, :, 2] = warped/255.
    
    
    cv.imshow("Fixed", img0)
    cv.imshow("Moving", img1)
    cv.imshow("Warped Moving", warped)
    cv.imshow("Diff", img)
    cv.waitKey()


  