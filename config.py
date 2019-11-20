import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

im_size = 128
batch_size = 16

num_samples = 30000
num_train = 26000
num_valid = 3000
num_test = 1000
image_folder = 'data/train2014'
train_file = 'data/train.pkl'
valid_file = 'data/valid.pkl'
test_file = 'data/test.pkl'

# Training parameters
num_workers = 1  # for data-loading
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none
