from dataloader import *
import argparse
import sys
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
import atexit
import matplotlib.pyplot as plt
from udacity_dataloader import UdacityDataset
import warnings
warnings.simplefilter(action='ignore')

train_split = 0.8
random_seed = 42
batch_size = 6
validation_split = 1-train_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Main function for clarifying the code
def main():
    args = arguments()
    model = pick_model(args.model)
    transformations = add_transforms(args)
    loader = create_loader(args, transformations)
    plot(args, loader)


# Function for generating the sample plots 
# and outputting the image file.
# Arguments:
# - dataloader: torch.dataloader
def plot(args, dataloader):
    # Generate samples for one batch of the loader
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            fig, axs = plt.subplots(int(batch_size/3), 3, figsize=(19.2,3.6*args.rows))
            images = sample_batched['image'].to(device)
            labels = sample_batched['steering_angle'].to(device).float()
            output = net(images.float())
            i = 0
            j = 0
            k = 0
            for image in images:
                axs[i, j].imshow(image.to('cpu').numpy().transpose((1, 2, 0)))
                axs[i, j].set_title("Prediction: {:.4f} \nLabel: {:.4f}".format(float(output[k]), float(labels[k])), fontsize=18)
                axs[i, j].axis('off')
                k += 1
                j += 1
                if j == 3:
                    j = 0
                    i += 1
            break
        fig.tight_layout()
        plt.savefig(args.sample_path)


# Function for defining argparser
def arguments():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model',
                            metavar='model',
                            type=str,
                            help='name of the model to validate',
                            default='PilotNet',
                            required=False)

    parser.add_argument('--sample_path',
                            metavar='sample_path',
                            type=str,
                            help='path to the sample image file',
                            default='samples.png',
                            required=False)

    parser.add_argument('--udacity',
                            help='use the Udacity dataset to generate the samples',
                            action='store_true')

    parser.add_argument('--quality',
                            help='define if the samples are generated from the best, worst or random data. Default=worst, options=random, best, worst',
                            metavar='quality',
                            type=str,
                            default='worst',
                            required=False)
    
    parser.add_argument('--augmented',
                            action='store_true',
                            help='define if the plotted samples are augmented with the training agmentations, default=False')

    parser.add_argument('--rows',
                            metavar='rows',
                            type=int,
                            help='number of 3 sample rows to be visualized',
                            default=2,
                            required=False)

    args = parser.parse_args()

    return args


# Function for choosing the model based on arguments:
# Currently only available model is PilotNet but
# supports easily adding new models as the rest of the code
def pick_model(name):
    model_names = ['PilotNet']

    if name not in model_names:
        print("Wrong model name, check the available models!")
        sys.exit()

    elif args.model == 'PilotNet':
        from models.PilotNet import PilotNet
        weight_file = 'weights/PilotNet_itlm_augs_10.pth'
        net = PilotNet().float().to(device)
        transformations = transforms.Compose([
                                            Crop((640, 1920)),
                                            Rescale((66, 200)),
                                            ToTensor(),
                                            #RandomLighting(),
                                            #RandomRotation(),
                                            #HorizontalFlip()
                                            ])
        
    else:
        print("Model should be available but not found")
        sys.exit()

    net.load_state_dict(torch.load(weight_file))
    net.eval()

    return net


# Define transformations based on the used data and/or augmentations
def add_transforms(args):
    if args.udacity and args.augmentations:
        transformations = transforms.Compose([
                                            Crop((213, 640), pixels_from_bottom=80),
                                            Rescale((66, 200)),
                                            ToTensor(),
                                            RandomLighting(),
                                            RandomRotation(),
                                            HorizontalFlip()
                                            ])
    elif args.udacity:
        transformations = transforms.Compose([
                                            Crop((213, 640), pixels_from_bottom=80),
                                            Rescale((66, 200)),
                                            ToTensor()
                                            ])
    elif args.augmentations:
        transformations = transforms.Compose([
                                            Crop((640, 1920)),
                                            Rescale((66, 200)),
                                            ToTensor(),
                                            RandomLighting(),
                                            RandomRotation(),
                                            HorizontalFlip()
                                            ])
    else:
        transformations = transforms.Compose([
                                            Crop((640, 1920)),
                                            Rescale((66, 200)),
                                            ToTensor()
                                            ])

    return transformations


# Create the dataloader for the specified set and transforms
def create_loader(args, transformations):
    batch_size = args.rows * 3 # Only one batch is generated which determines the number of visualized samples
    if args.udacity:
        data = UdacityDataset('udacity_data/steering.csv', 'udacity_data/center', transform=transformations)
    else:
        data = ImgSteeringDataset(bus_file='data/camera_lidar/20180810_150607/bus/20180810150607_bus_signals.json', img_dir='data/camera_lidar/20180810_150607/camera/cam_front_center/', transform=transformations)
    
    # Train/validation split in the same way as in the training script
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    train_indices, val_indices = indices[:-split], indices[-split:]
    
    if args.data == 'validation':
        dataset = torch.utils.data.Subset(data, val_indices)
    else:
        dataset = torch.utils.data.Subset(data, train_indices)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # if the best or worst samples are visualized, 
    # create a smaller loader from sorted samples
    if args.quality != 'random':
        with torch.no_grad():
        sample_losses = {}
        criterion = nn.MSELoss(reduction='none')
        for itlm_batch, sample in enumerate(train_loader):
            with torch.no_grad():
                images = sample['image'].to(device)
                labels = sample['steering_angle'].to(device).float()
                itlm_losses = criterion(net(images.float()), labels.view(labels.shape[0], 1)).float()
                i = 0
                for itlm_loss in itlm_losses:
                    sample_losses[itlm_batch*batch_size+i] = itlm_loss
                    i += 1
        # Trim the data based on highest loss values:
        # Sorting from highest loss onwards if the worst samples are used
        worst = (args.quality == 'worst')
        sorted_sample_losses = {k: v for k, v in sorted(sample_losses.items(), key=lambda item: item[1], reverse=worst)}
        full_length = len(sample_losses)
        subsample_indices = islice(sorted_sample_losses.items(), int(full_length * (1 - 0.98))) # 2% of the top samples
        # Every third element from the keys to switch to 10 fps
        dataset = torch.utils.data.Subset(data, np.array(list(subsample_indices))[:,0]*3) 
        data_transformed = MapDataset(train_set, transformations) # Add training augmentations
        loader = DataLoader(training_data_transformed, batch_size=batch_size, shuffle=True)

    return 

main()