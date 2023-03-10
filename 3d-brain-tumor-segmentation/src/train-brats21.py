import argparse
import json
import logging
import os
import time
import random

import matplotlib.pyplot as plt
import mlflow
import torch
from torch.utils.tensorboard import SummaryWriter

# MONAI imports
from monai.config import print_config
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations, AsDiscrete, Compose, ConcatItemsd, EnsureChannelFirstd, EnsureTyped, LoadImaged, MapTransform, NormalizeIntensityd,
    Orientationd, RandFlipd, RandScaleIntensityd, RandShiftIntensityd, RandSpatialCropd, Spacingd,
    )
from monai.utils import set_determinism


# Avoid flooding of debug messages in logs
logging.basicConfig(level=logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("azureml").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("azure.mlflow").setLevel(logging.WARNING)

# MONAI config
print_config()

start_run = time.time()

# SET CENTRAL VARIABLES

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str, help="path to input data")
parser.add_argument("--epochs", type=int, default=50, help="no of epochs")
parser.add_argument("--initial_lr", type=float, default=0.0001, help="Initial learning rate")
parser.add_argument("--best_model_name", type=str, default='best-model', help="Name of best model to register in AzureML")
parser.add_argument("--train_batch_size", type=int, default=1, help="Train loader batch size")
parser.add_argument("--val_batch_size", type=int, default=1, help="Validation loader batch size")

args = parser.parse_args()

max_epochs = args.epochs
initial_lr = args.initial_lr
best_model_name = args.best_model_name
train_batch_size = args.train_batch_size
val_batch_size = args.val_batch_size
input_data_dir = args.input_data

# Select logging targets for metrics. tb for tensorboard and/or mlflow
log_targets = ['tb', 'mlflow']

# AzureML job asset folder. Will be used to store model checkpoints
azureml_output_folder = './outputs'

datalist_json_path = os.path.join(input_data_dir, 'dataset.json')
VAL_AMP = True # MONAI validation mixed precision

# Distributed training:    
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank) if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(device)
torch.distributed.init_process_group(backend="nccl", init_method="env://")
world_size = torch.distributed.get_world_size()

# Init logging for AzureML

if 'tb' in log_targets:
    tb_writer = SummaryWriter("./tb_logs")

# init mlflow also if not in log_targets because it is needed to register the model
mlflow.autolog(silent=True)

params = {
    "Epochs": max_epochs,
    "Initial lr" : initial_lr,
    "Train batch size" : train_batch_size,
    "Validation batch size" : val_batch_size,
    "Register best model as" : best_model_name,
    "Val_auto_mixed_prec" : VAL_AMP
}

# rank == 0 to let only one GPU worker perform the operation
if rank == 0 and 'mlflow' in log_targets:
    try:
        mlflow.log_params(params)
    except Exception as e:
        print('Exception during mlflow parameter logging: {e}')

# Set deterministic training for reproducibility
set_determinism(seed=0)

# Custom transform to convert the multi-classes labels into multi-labels segmentation task in One-Hot format.
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats 2021 classes:
    label 1 necrotic tumor core (NCR)
    label 2 peritumoral edematous/invaded tissue 
    label 3 is not used in the new dataset version
    label 4 GD-enhancing tumor 
    The possible classes are:
      TC (Tumor core): merge labels 1 and 4
      WT (Whole tumor): merge labels 1,2 and 4
      ET (Enhancing tumor): label 4

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 1 and label 4 to construct TC
            result.append(torch.logical_or(d[key] == 1, d[key] == 4))
            # merge labels 1, 2 and 4 to construct WT
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 1, d[key] == 2), d[key] == 4
                )
            )
            # label 4 is ET
            result.append(d[key] == 4)
            d[key] = torch.stack(result, axis=0).float()
        return d

# Generate training and validation input lists for dataloaders

image_folders = [f for f in os.listdir(input_data_dir) if os.path.isdir(os.path.join(input_data_dir, f))]
print(f'{len(image_folders)} images in image_folders.')

print(image_folders[:5])

train_frac = 0.8

train_size = int(0.8 * len(image_folders))

train_folders = image_folders[:train_size]
val_folders = image_folders[train_size:]

def create_datalist(folders):

    elements = []
    for folder in folders:

        folder_path = os.path.join(input_data_dir, folder)

        flair_file = next((f for f in os.listdir(folder_path) if f.endswith('flair.nii.gz')), None)
        t1_file = next((f for f in os.listdir(folder_path) if f.endswith('t1.nii.gz')), None)
        t1ce_file = next((f for f in os.listdir(folder_path) if f.endswith('t1ce.nii.gz')), None)
        t2_file = next((f for f in os.listdir(folder_path) if f.endswith('t2.nii.gz')), None)
        label_file = next((f for f in os.listdir(folder_path) if f.endswith('seg.nii.gz')), None)
        
        element = {
            'flair' : os.path.join(folder_path, flair_file),
            't1' : os.path.join(folder_path, t1_file),
            't1ce' : os.path.join(folder_path, t1ce_file),
            't2' : os.path.join(folder_path, t2_file),
            'label' : os.path.join(folder_path, label_file),
        }
        elements.append(element)

    return elements

train_list = create_datalist(train_folders)
valid_list = create_datalist(val_folders)

print(f'{len(train_list)} training images and {len(valid_list)} validation images found.')


# Setup transforms for training and validation

train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["flair", "t1", "t1ce", "t2", "label"]),
        EnsureChannelFirstd(keys=["flair", "t1", "t1ce", "t2"]),
        ConcatItemsd(keys=["flair", "t1", "t1ce", "t2"], name="image", dim=0),
        
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=["flair", "t1", "t1ce", "t2", "label"]),
        EnsureChannelFirstd(keys=["flair", "t1", "t1ce", "t2"]),
        ConcatItemsd(keys=["flair", "t1", "t1ce", "t2"], name="image", dim=0),
                
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

# Load data
# adjust caching rates in case of out of memory issue

train_ds = Dataset(data=train_list, transform=train_transform)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
train_loader = DataLoader(train_ds, batch_size= train_batch_size, shuffle=False, num_workers=12, sampler=train_sampler)

val_ds = Dataset(data=valid_list, transform=val_transform)
val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=12)

params = {
    "train_samples": len(train_ds),
    "val_samples" : len(val_ds)
}

if rank == 0 and 'mlflow' in log_targets:
    try:
        mlflow.log_params(params)
    except Exception as e:
        print('Exception during mlflow parameter logging: {e}')

    # log sample images
    val_data_example = val_ds[2]
    print(f"image shape: {val_data_example['image'].shape}")
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    for i in range(4):
        axs[0, i].set_title(f"image channel {i}")
        axs[0, i].imshow(val_data_example["image"][i, :, :, 60].detach().cpu(), cmap="gray")

    # also visualize the 3 channels label corresponding to this image
    print(f"label shape: {val_data_example['label'].shape}")

    for i in range(3):
        axs[1, i].set_title(f"label channel {i}")
        axs[1, i].imshow(val_data_example["label"][i, :, :, 60].detach().cpu())

    # add an empty subplot to align the last label image with the others
    axs[1, 3].axis('off')

    # plt.show()   
    samples = plt.gcf()
    try:
        mlflow.log_figure(samples, 'sample-images.png')
    except Exception as e:
        print('Exception during mlflow image logging: {e}')

# Create model, loss and optimizer

val_interval = 1

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.nn.parallel.DistributedDataParallel(module= SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
).to(device), device_ids=[local_rank])

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), initial_lr, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

# define inference method
def inference(input):

    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

start_training = time.time()

preprocessing_mins = (start_training - start_run) / 60

if 'mlflow' in log_targets:
    try:
        mlflow.log_metric("preprocessing_mins", preprocessing_mins, 0)
    except Exception as e:
        print(f'Exception occured writing train metrics to mlflow: {e}')

# RUN TRAINING

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []

total_start = time.time()

for epoch in range(max_epochs):
    epoch_start = time.time()
    train_sampler.set_epoch(epoch) # parallel training
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
 
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    # Log epoch train metrics
    if 'tb' in log_targets:
        try:
            tb_writer.add_scalar("train_loss", epoch_loss, epoch+1)
        except Exception as e:
            print(f'Exception occured writing train metrics to tensorboard: {e}')

    if 'mlflow' in log_targets:
        try:
            mlflow.log_metric("train_loss", epoch_loss, epoch+1)
        except Exception as e:
            print(f'Exception occured writing train metrics to mlflow: {e}')
    
    # VALIDATION
    if (epoch + 1) % val_interval == 0:
        val_epoch_loss = 0
        val_step = 0     
        model.eval()
        with torch.no_grad():

            for val_data in val_loader:
                val_step += 1
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = inference(val_inputs)
                val_loss = loss_function(val_outputs, val_labels)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

                val_epoch_loss += val_loss.item()

            val_epoch_loss /= val_step

            metric = dice_metric.aggregate().item()
            #metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            #metric_values_tc.append(metric_tc)
            metric_wt = metric_batch[1].item()
            #metric_values_wt.append(metric_wt)
            metric_et = metric_batch[2].item()
            #metric_values_et.append(metric_et)
            dice_metric.reset()
            dice_metric_batch.reset()

            # Let one instance 0 save model if validation loss improved
            if rank == 0:
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1

                    model_checkpoint = os.path.join(azureml_output_folder, "best_metric_model.pth")
                    torch.save(model.module.state_dict(), model_checkpoint)

                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )

            epoch_duration_s = time.time() - epoch_start
    
            # Log epoch val metrics:
            
            epoch_val_metrics = {
                'val_loss' : val_epoch_loss,
                'val_mean_dice' : metric,
                'val_dice_tc' : metric_tc,
                'val_dice_wt' : metric_wt,
                'val_dice_et' : metric_et,
                'epoch_duration_s' : epoch_duration_s
            }

            if 'tb' in log_targets:
                try:
                    for name, value in epoch_val_metrics.items():
                        tb_writer.add_scalar(tag= name, scalar_value= value, global_step= epoch+1)
                except:
                    print(f'Exception occured writing validation metrics to tensorboard: {e}')
            if 'mlflow' in log_targets:
                try:
                    mlflow.log_metrics(metrics= epoch_val_metrics, step= epoch+1)
                except:
                    print(f'Exception occured writing validation metrics to MLFLow: {e}')

    print(f"time consuming of epoch {epoch + 1} is: {epoch_duration_s:.4f}")
    
total_time_mins = (time.time() - total_start) / 60
if rank == 0:    
    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch},"
        f"total time: {total_time_mins:.2f} mins ({total_time_mins/max_epochs:.2f} per epoch).")

    # Load the best model into memory
    
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)
        
    model.load_state_dict(torch.load(model_checkpoint))

    print("Registering the model via MLFlow")
    
    try:
        mlflow.pytorch.log_model(
            pytorch_model=model,
            registered_model_name= best_model_name,
            artifact_path= 'model',
            extra_files=[model_checkpoint])
        
    except Exception as e:
        print(e)

# End mlflow run and tb writer
if mlflow.active_run():
    mlflow.end_run()

if tb_writer is not None:
    tb_writer.close()