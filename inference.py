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

    top_point = (rho, rho)
    left_point = (top_point[0], patch_size + top_point[1])
    bottom_point = (patch_size + top_point[0], patch_size + top_point[1])
    right_point = (patch_size + top_point[0], top_point[1])

    four_points = [top_point, left_point, bottom_point, right_point]

    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))
    
    H = cv.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = inv(H)

    warped_image = cv.warpPerspective(Image, H_inverse, size)

    Ip1 = Image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
    Ip2 = warped_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]

    return Ip1, Ip2



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

    
    img0, img1 = data_maker("C:/Users/xwen2/Desktop/HomographyNet_supervised/inference/5.jpg")
    
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


  