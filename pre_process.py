import os
import pickle
import random

import cv2 as cv
import numpy as np
from numpy.linalg import inv
from tqdm import tqdm

from config import image_folder
from config import train_file, valid_file, test_file


def get_datum(img, test_image, img_perturbed, size, rho, top_point, patch_size):
    left_point = (top_point[0], patch_size + top_point[1])
    bottom_point = (patch_size + top_point[0], patch_size + top_point[1])
    right_point = (patch_size + top_point[0], top_point[1])
    four_points = [top_point, left_point, bottom_point, right_point]

    perturbed_four_points = []
    flag = 0
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, +rho), point[1] + random.randint(-rho, +rho)))
        '''
        if (flag == 0):
            perturbed_four_points.append((point[0] + random.randint(-rho, 0), point[1] + random.randint(-rho, 0)))
        elif(flag == 1):
            perturbed_four_points.append((point[0] + random.randint(-rho, 0), point[1] + random.randint(0, +rho)))
        elif(flag == 2):
            perturbed_four_points.append((point[0] + random.randint(0, +rho), point[1] + random.randint(0, +rho)))
        elif(flag == 3):
            perturbed_four_points.append((point[0] + random.randint(0, +rho), point[1] + random.randint(-rho, 0)))
        flag = flag + 1
        '''

    H = cv.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    H_inverse = inv(H)

    warped_image = cv.warpPerspective(img_perturbed, H_inverse, size)

    Ip1 = test_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]
    Ip2 = warped_image[top_point[1]:bottom_point[1], top_point[0]:bottom_point[0]]

    training_image = np.dstack((Ip1, Ip2))
    datum = (training_image, np.array(four_points), np.array(perturbed_four_points))
    return datum

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
    degree_ = 18
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

    kernel = [random.randint(1, 3) * 2 + 1 for x in range(1)]
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


def process(files, is_test):
    if is_test:
        size = (640, 480)
        rho = 80
        patch_size = 256

    else:
        size = (320, 240)
        rho = 40
        patch_size = 128

    samples = []
    
    for f in tqdm(files):
        fullpath = os.path.join(image_folder, f)
        img = cv.imread(fullpath, 0)
        img = cv.resize(img, size)
        test_image = img.copy()

        img_perturbed= cv.imread(fullpath, 1)
        img_perturbed = perturbed_image(img_perturbed)
        img_perturbed = cv.cvtColor(img_perturbed, cv.COLOR_BGR2GRAY)
        img_perturbed = cv.resize(img_perturbed, size)

        if not is_test:
            for top_point in [(0 + 32, 0 + 32), (128 + 32, 0 + 32), (0 + 32, 48 + 32), (128 + 32, 48 + 32),
                              (64 + 32, 24 + 32)]:
                datum = get_datum(img, test_image, img_perturbed, size, rho, top_point, patch_size)
                samples.append(datum)
        else:
            top_point = (rho, rho)
            datum = get_datum(img, test_image, img_perturbed, size, rho, top_point, patch_size)
            samples.append(datum)

    return samples


if __name__ == "__main__":
    files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
    np.random.shuffle(files)

    num_files = len(files)
    print('num_files: ' + str(num_files))

    num_train_files = 26000
    num_valid_files = 3000
    num_test_files = 1000

    train_files = files[:num_train_files]
    valid_files = files[num_train_files:num_train_files + num_valid_files]
    test_files = files[num_train_files + num_valid_files:num_train_files + num_valid_files + num_test_files]

    train = process(train_files, False)
    valid = process(valid_files, False)
    test = process(test_files, True)

    print('num_train: ' + str(len(train)))
    print('num_valid: ' + str(len(valid)))
    print('num_test:  ' + str(len(test)))

    with open(train_file, 'wb') as f:
        pickle.dump(train, f)  # serialize object into file
    with open(valid_file, 'wb') as f:
        pickle.dump(valid, f)
    with open(test_file, 'wb') as f:
        pickle.dump(test, f)
