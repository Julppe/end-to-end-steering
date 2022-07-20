from dataloader import *
import argparse
import sys
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
import atexit
import skimage

train_split = 0.8
random_seed = 42
batch_size = 32

validation_split = 1-train_split
# Transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Argparse model name
model_names = ['SimpleCNN', 'PilotNet']
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('model',
                        metavar='model',
                        type=str,
                        help='name of the model to validate',
                        default='SimpleCNN')

args = parser.parse_args()
if args.model not in model_names:
    print("Wrong model name, check the available models!")
    sys.exit()

elif args.model == 'SimpleCNN':
    from models.SimpleCNN import SimpleCNN
    weight_file = 'SimpleCNN.pth'
    net = SimpleCNN().float().to(device)
    transformations = transforms.Compose([ToTensor()])

elif args.model == 'PilotNet':
    from models.PilotNet import PilotNet
    weight_file = 'weights/PilotNet_itlm_augs_10.pth'
    net = PilotNet().float().to(device)
    transformations = transforms.Compose([Crop((640, 1920)), Rescale((66, 200)), ToTensor()])
    
else:
    print("Model should be available but not found")
    sys.exit()

net.load_state_dict(torch.load(weight_file))
net.eval()

data = ImgSteeringDataset(bus_file='data/camera_lidar/20180810_150607/bus/20180810150607_bus_signals.json', img_dir='data/camera_lidar/20180810_150607/camera/cam_front_center/', transform=transformations)

# Train/validation split
dataset_size = len(data)
indices = list(range(dataset_size))
split = int(np.floor(train_split * dataset_size))
val_indices, train_indices = indices[split:], indices[:split]

train_dataset = torch.utils.data.Subset(data, train_indices)
valid_dataset = torch.utils.data.Subset(data, val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


valid_criterion = nn.L1Loss() # Use MAE Loss to give a proper estimate about the size of the steering error

with torch.no_grad():
    total_loss = 0
    print("Starting the validation loop")
    print("----------------------------")
    for i_batch, sample_batched in enumerate(valid_loader):
        images = sample_batched['image'].to(device)
        labels = sample_batched['steering_angle'].to(device).float()
        output = net(images.float())
        loss = valid_criterion(output, labels.view(labels.shape[0], 1)).float()
        total_loss += loss.item()

    avg_loss = total_loss/(i_batch+1)

print("Validation loop finished. Validation error computed with {:d} images.".format((i_batch+1)*batch_size))
print("---------------------------------------------------------------------")
print("Average MAELoss: {:.7f}".format(avg_loss))
print("In degrees of steering angle: {:.7f}".format(avg_loss/np.pi * 180))
