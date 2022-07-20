from dataloader import *
import argparse
import sys
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
import atexit
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import islice
import pandas as pd


# Save model parameters in case of a generic error or keyboard interrupt
def exit_handler():
    print("Saving weights to file: " + weight_file)
    torch.save(net.to('cpu').state_dict(), weight_file)
    print("Saving errors to CSV: " + csv_file)
    error_df.to_csv(csv_file, sep=';')

atexit.register(exit_handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: ' + str(device))
# Argparse for model name, ITLM, augmentations
model_names = ['PilotNet']
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('model',
                        metavar='model',
                        type=str,
                        help='name of the model to train',
                        default='SimpleCNN')
parser.add_argument('--no_itlm',
                        help='define if the itlm algorithm is not used in training',
                        action='store_true')
parser.add_argument('--no_augs',
                        help='Define if data augmentations are not used',
                        action='store_true')
parser.add_argument('--alpha',
                        metavar='alpha',
                        help='define alpha value used for the ITLM algorithm',
                        type=float,
                        default=0.05,
                        required=False)
parser.add_argument('--image_path',
                        metavar='image_path',
                        type=str,
                        help='path of the image folder',
                        default='data/camera_lidar/20180810_150607/camera/cam_front_center/')
parser.add_argument('')


args = parser.parse_args()
if not (0 <= args.alpha < 1):
    print("Alpha must be in the range [0, 1]")
    sys.exit()
itlm_alpha = args.alpha   # base value -5% samples each epoch

if args.model not in model_names:
    print("Wrong model name, check the available models!")
    sys.exit()

elif args.model == 'PilotNet':
    from models.PilotNet import PilotNet
    weight_file = 'PilotNet_itlm.pth'
    net = PilotNet().float().to(device)
    if not args.no_augs:
        training_transformations = transforms.Compose([
                                        Crop((640, 1920)),
                                        Rescale((66, 200)),
                                        ToTensor(),
                                        RandomLighting(),
                                        RandomRotation(),
                                        HorizontalFlip()
                                        ])
    else:
        training_transformations = transforms.Compose([
                                        Crop((640, 1920)),
                                        Rescale((66, 200)),
                                        ToTensor()
                                        ])
    validation_transformations = transforms.Compose([
                                        Crop((640, 1920)),
                                        Rescale((66, 200)),
                                        ToTensor()
                                        ])
    itlm_transformations = validation_transformations
    
else:
    print("Model should be available but not found")
    sys.exit()

# Define the names of the plot and weight outputs based on the model arguments
output_file_name = args.model
if not args.no_itlm:
    output_file_name = output_file_name + '_itlm'
else:
    output_file_name = output_file_name + '_no_itlm'
if not args.no_augs:
    output_file_name = output_file_name + '_augs'
else:
    output_file_name = output_file_name + '_no_augs'
if not args.no_itlm:
    output_file_name = output_file_name + '_' + str(int(itlm_alpha*100))

plot_file = 'outputs'+'/'+ output_file_name + '.png'
csv_file = 'outputs'+'/'+ output_file_name + '.csv'
weight_file = output_file_name + '.pth'

train_split = 0.8
random_seed = 42
batch_size = 32

validation_split = 1-train_split
# Transforms

# Because of the way the dataloader is built, the initialisation takes quite long
# but this makes every iteration of the training loop faster reducing total training time
data = ImgSteeringDataset(bus_file='data/camera_lidar/20180810_150607/bus/20180810150607_bus_signals.json', img_dir=args.image_path)

# 80/20 random train-validation split
dataset_size = len(data)
# Non random train-validation split, for better generalisation evaluation
# Only use every third image for both sets to change from 30 fps to 10 fps
indices = list(range(0, dataset_size, 3))
split = int(np.floor((validation_split * dataset_size))/3)
train_indices, val_indices = indices[split:], indices[:split]
train_dataset = torch.utils.data.Subset(data, train_indices)
valid_dataset = torch.utils.data.Subset(data, val_indices)
itlm_dataset = train_dataset

# Validation is done without the data augmentations
training_data_transformed = MapDataset(train_dataset, training_transformations)
validation_data_transformed = MapDataset(valid_dataset, validation_transformations)
itlm_data_transformed = MapDataset(itlm_dataset, itlm_transformations)

train_loader = DataLoader(training_data_transformed, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(validation_data_transformed, batch_size=batch_size, shuffle=True)
itlm_loader = DataLoader(itlm_data_transformed, batch_size=batch_size, shuffle=False)


# Training loop
optimizer = optim.Adam(net.parameters(), lr=0.0001)
criterion = nn.MSELoss()
parallel_valid_criterion = nn.MSELoss()
epochs = 20
# Lists of errors for plotting purposes
train_losses = []
valid_losses = []
x = []
train_loss_mms = []
valid_loss_mms = []
# Dataframe for saving all the losses to a CSV-file for post-training plots and analysis
error_df = pd.DataFrame(columns=['epoch', 'batch', 'training_error', 'validation_error'])
first_loader_length = len(train_loader)

print("Starting the training loop")
print("--------------------------")
for epoch in range(epochs):
    subset_loss = 0
    batch_loss = 0
    valid_loss = 0
    # Generate new training subset using ITLM from the second epoch onwards
    if epoch >= 1 and not args.no_itlm:
        print("-------------------------------------")
        print("Generating new trainloader using ITLM")
        print("-------------------------------------")
        sample_losses = {}
        itlm_criterion = nn.MSELoss(reduction='none')
        # Set net.eval() because the net is only used to generate the losses for ITLM
        net.eval()
        # ITLM loader is the same as train loader with no augmentations and full 30 fps image set
        # The new trainloader is generated by sorting the indices of the losses and creating a 
        # new dataloader using the lowest loss indices from the training set.
        for itlm_batch, sample in enumerate(itlm_loader):
            with torch.no_grad():
                images = sample['image'].to(device)
                labels = sample['steering_angle'].to(device).float()
                itlm_losses = itlm_criterion(net(images.float()), labels.view(labels.shape[0], 1)).float()
                i = 0
                for itlm_loss in itlm_losses:
                    sample_losses[itlm_batch*batch_size+i] = itlm_loss
                    i += 1
        # Trim the data based on highest loss values:
        sorted_sample_losses = {k: v for k, v in sorted(sample_losses.items(), key=lambda item: item[1])}
        full_length = len(sample_losses)
        subsample_indices = islice(sorted_sample_losses.items(), int(full_length * (1 - itlm_alpha)))
        train_set = torch.utils.data.Subset(data, np.array(list(subsample_indices))[:,0]*3 + len(val_indices)*3) # Every third element from the keys to switch to 10 fps
        training_data_transformed = MapDataset(train_set, training_transformations) # Add training augmentations
        train_loader = DataLoader(training_data_transformed, batch_size=batch_size, shuffle=True)
        print("Generated new trainloader with {:d} samples".format(len(train_set)))
        print("-------------------------------------------")

    print("Starting epoch {:d}:".format(epoch))
    # The training loop
    for i_batch in range(len(train_loader)):
        loss = 0
        net.train()
        optimizer.zero_grad()
        sample_batched = next(iter(train_loader))
        images = sample_batched['image'].to(device)
        labels = sample_batched['steering_angle'].to(device).float()
        output = net(images.float())
        loss = criterion(output, labels.view(labels.shape[0], 1)).float()
        loss_item = loss.item()
        loss.backward()
        optimizer.step()
        batch_loss += loss_item
        subset_loss += loss_item
        train_losses.append(loss_item)

        net.eval()
        with torch.no_grad():
            sample_valid = next(iter(valid_loader))
            valid_images = sample_valid['image'].to(device)
            valid_labels = sample_valid['steering_angle'].to(device).float()
            valid_output = net(valid_images.float())
            valid_loss = parallel_valid_criterion(valid_output, valid_labels.view(valid_labels.shape[0], 1)).float().item()
            valid_losses.append(valid_loss)

        error_df.loc[len(error_df.index)] = [epoch, i_batch, loss_item, valid_loss]
        if i_batch >= 20 or epoch >= 1:
            # Plot moving average over 20 batches 
            # validation vs training error.
            # Print and plot both errors.
            train_loss_mms.append(sum(train_losses[-20:])/20)
            valid_loss_mms.append(sum(valid_losses[-20:])/20)
            # X axis value as number of batches so far
            if epoch >= 1:
                x.append(first_loader_length + len(train_loader)*(epoch-1) + i_batch)
            else:
                x.append(i_batch)
            
            # Generate and save the plot
            plt.plot(x, train_loss_mms, label='Training error', color='blue')
            plt.plot(x, valid_loss_mms, label='Validation error', color='gold')
            plt.xlabel('Number of batches')
            plt.ylabel('MSE loss')
            plt.title('Training and validation moving average over 20 batches')
            plt.savefig(plot_file)

        # Print errors only every 20 batches    
        if i_batch % 20 == 0 and i_batch != 0:
            # Print errors to terminal
            print("Epoch {:d}, i_batch: {:d}".format(epoch, i_batch))
            print("Moving average training error: {:f}".format(train_loss_mms[-1]))
            print("Moving average validation error: {:f}".format(valid_loss_mms[-1]))
            print("Read {:d} files".format(int(batch_size*(i_batch+1))))
            print("--------------------------")
            subset_loss = 0
            valid_loss = 0

    # After each epoch print the last average errors 
    # and number of files in the epoch
    print("Epoch {:d} finished. Final numbers:".format(epoch))
    print("Last moving average training error: {:f}".format(train_loss_mms[-1]))
    print("Last moving average validation error: {:f}".format(valid_loss_mms[-1]))
    print("Read {:d} files".format(int(batch_size*(i_batch+1))))
    print("---------------")

print("----------------------------------------------------------")
print("Training complete, last epoch average training error: {:f}".format(batch_loss/(i_batch+1)))
print("----------------------------------------------------------")

print("Saving weights to file: " + weight_file)
torch.save(net.state_dict(),weight_file)
print("Saving errors to CSV: " + csv_file)
error_df.to_csv(csv_file, sep=';')

# Compute the complete validation accuracy after the end of training
valid_criterion = nn.L1Loss()
net.eval()
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
print("In degrees of steering angle: {:.7f}".format((avg_loss/np.pi) * 180))
