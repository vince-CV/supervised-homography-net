import time
import cv2 as cv
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from config import batch_size, num_workers
from data_gen import DeepHNDataset
from mobilenet_v2 import MobileNetV2
from utils import AverageMeter

device = torch.device('cpu')


if __name__ == '__main__':
    
    filename = 'model/homonet.pt'
    print('loading {}...'.format(filename))
    model = MobileNetV2()
    model.load_state_dict(torch.load(filename))
    model.eval()

    imsize = 128 if torch.cuda.is_available() else 128  # use small size if no gpu

    
    img0 = cv.imread("C:/Users/xwen2/Desktop/inference/src.jpg", 0)
    img1 = cv.imread("C:/Users/xwen2/Desktop/inference/dst.jpg", 0)

    img0 = cv.resize(img0, (128, 128))
    img1 = cv.resize(img1, (128, 128))

    img = np.zeros((128, 128, 3), np.float32)
    img[:, :, 0] = img0 / 255.
    img[:, :, 1] = img1 / 255.
    #img = np.transpose(img, (1, 0, 2))  # HxWxC array to CxHxW

    transformation = transforms.Compose([transforms.ToTensor(), ])   # covert np.array into tensor
    img_Tensor = transformation(img)
    img_Tensor=torch.unsqueeze(img_Tensor, 0)  # append extra dimention on the tensor

    #diff = [(0,0), (0,0), (0,0), (0,0)]

    #test_dataset = DeepHNDataset('test')
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=num_workers)

    ##img = torch.utils.data.DataLoader(img, batch_size = 1, shuffle=False, num_workers=num_workers)
    img_Tensor = img_Tensor.to(device)  # [N, 3, 128, 128]
    
    #target = target.float().to(device)  # [N, 8]

    with torch.no_grad():
        #start = time.time()
        out = model(img_Tensor)  # [N, 8]
        #end = time.time()
        #elapsed = elapsed + (end - start)

        # Calculate loss

    print("Output: \n", out)
  