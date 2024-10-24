import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob  # Filename pattern matching
import time
import cv2
import matplotlib.patches as mpatches
from scipy.ndimage.morphology import binary_dilation
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import albumentations as A
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from scipy.ndimage import binary_dilation
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device)) 
def generate_file_paths(path):
    file_base, extension = os.path.splitext(path)
    file_name = os.path.basename(path)
    return [path, f'{file_base}_mask{extension}']

data_directory = 'D:/cancerSegmentation/mri_segmentation/kaggle_3m/'
image_paths = glob(f'{data_directory}/*/*[0-9].tif')

# Create a DataFrame with image and mask file paths
file_paths_df = pd.DataFrame((generate_file_paths(file_path) for file_path in image_paths), columns=['image_path', 'mask_path'])

class BrainDataset(Dataset):
    def __init__(self, data_frame, transformations=None, mean_value=0.5, std_value=0.25):
        super(BrainDataset, self).__init__()
        self.data_frame = data_frame
        self.transformations = transformations
        self.mean_value = mean_value
        self.std_value = std_value

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index, raw=False):
        row_data = self.data_frame.iloc[index]
        image = cv2.imread(row_data['image_path'], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(row_data['mask_path'], cv2.IMREAD_GRAYSCALE)
        if raw:
            return image, mask

        if self.transformations:
            augmented = self.transformations(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        image = T.functional.to_tensor(image)
        mask = mask // 255
        mask = torch.Tensor(mask)
        return image, mask

# Data splitting

train_data, test_data = train_test_split(file_paths_df, test_size=0.2)

augmentation = A.Compose([
    A.ChannelDropout(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.ColorJitter(p=0.3),
])

# Data loaders
# Create datasets and data loaders for training, validation, and testing.

train_dataset = BrainDataset(train_data, transformations=augmentation)
test_dataset = BrainDataset(test_data)
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

def calculate_iou(predictions, targets, epsilon=1e-7):
    predictions = torch.where(predictions > 0.5, 1, 0)
    targets = targets.byte()
    intersection = (predictions & targets).float().sum((1, 2))
    union = (predictions | targets).float().sum((1, 2))
    iou = (intersection + epsilon) / (union + epsilon)
    return iou

def calculate_dice(predictions, targets, epsilon=1e-7):
    predictions = torch.where(predictions > 0.5, 1, 0)
    targets = targets.byte()
    intersection = (predictions & targets).float().sum((1, 2))
    return ((2 * intersection) + epsilon) / (predictions.float().sum((1, 2)) + targets.float().sum((1, 2)) + epsilon)

def custom_loss_function(output, target, alpha=0.01):
    bce_loss = torch.nn.functional.binary_cross_entropy(output, target)
    soft_dice_loss = 1 - calculate_dice(output, target).mean()
    return bce_loss + alpha * soft_dice_loss

from models.EDRNet import EDRNet
model=EDRNet(num_classes=1)
model.to(device)
checkpoint_dir = 'D:/cancerSegmentation/BCS_ad10_EFFEMRI'

checkpoint_file = 'best_model.pt'
model_path = os.path.join(checkpoint_dir, checkpoint_file)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f'Model weights loaded from epoch: {model_path}')
else:
    print("Checkpoint for epoch  does not exist.")

loss_function = custom_loss_function
optimizer = Adam(model.parameters(), lr=0.001)
num_epochs = 100
learning_rate_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=2, factor=0.2)

def compute_iou(prediction, mask):
    prediction = prediction.int()
    mask = mask.int()
    intersection = (prediction & mask).float().sum()
    union = (prediction | mask).float().sum()
    iou_score = (intersection + 1e-7) / (union + 1e-7)
    return iou_score.item()




def display_images_and_masks(images, masks, pred_masks, scores, num_images=4):
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    # fig.suptitle('Original Image | Original Mask | Predicted Mask', fontsize=16)

    for i in range(num_images):
        # Move tensors to CPU and convert to numpy
        image = images[i].cpu().numpy()
        mask = masks[i].cpu().numpy()
        pred_mask = pred_masks[i].cpu().numpy()

        # Reduce brightness of the background image
        dim_factor = 0.3
        dimmed_image = np.clip(image * dim_factor, 0, 1)

        # Original Image
        axes[i, 0].imshow(np.transpose(image, (1, 2, 0)))
        axes[i, 0].axis('off')

        # Original Mask on Original Image
        img_with_mask = np.transpose(dimmed_image, (1, 2, 0)).copy()
        yellow_mask = np.zeros_like(img_with_mask)
        yellow_mask[mask.astype(bool)] = [1, 1, 0]  # Yellow color for mask area
        img_with_mask = np.clip(img_with_mask + yellow_mask * 0.5, 0, 1)  # Blend mask with image
        axes[i, 1].imshow(img_with_mask)
        axes[i, 1].axis('off')

        # Predicted Mask on Original Image
        img_with_pred = np.transpose(dimmed_image, (1, 2, 0)).copy()
        red_mask = np.zeros_like(img_with_pred)
        red_mask[pred_mask.astype(bool)] = [1, 0, 0]  # Red color for predicted mask area
        img_with_pred = np.clip(img_with_pred + red_mask * 0.5, 0, 1)  # Blend mask with image
        axes[i, 2].imshow(img_with_pred)
        axes[i, 2].axis('off')

        # Add score to predicted mask
        axes[i, 2].text(10, 20, f'IoU: {scores[i]:.4f}', color='white', fontsize=10,
                        bbox=dict(facecolor='blue', alpha=0.5))


    plt.tight_layout()
    plt.show()


num_examples = 4
images, masks, pred_masks, scores = [], [], [], []

with torch.no_grad():
    for batch in test_loader:
        image, mask = batch
        mask = mask[0]
        if not mask.byte().any():
            continue
        image = image.to(device)
        prediction = model(image).to('cpu')[0][0]
        prediction = torch.where(prediction > 0.5, 1, 0)
        iou_score = compute_iou(prediction, mask)

        images.append(image[0].cpu())  # Move to CPU here
        masks.append(mask.cpu())  # Move to CPU here
        pred_masks.append(prediction.cpu())  # Move to CPU here
        scores.append(iou_score)

        if len(images) == num_examples:
            break

display_images_and_masks(images, masks, pred_masks, scores)


from sklearn.metrics import accuracy_score, confusion_matrix

def calculate_metrics(pred, target):
    prediction = torch.where(pred > 0.5, 1, 0)
    dice = calculate_dice(prediction, target)
    iou = compute_iou(prediction, target)


    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (target >= 0.5).float()
    gt_binary_inverse = (gt_binary == 0).float()

    tp = pred_binary.mul(gt_binary).sum()
    fp = pred_binary.mul(gt_binary_inverse).sum()
    tn = pred_binary_inverse.mul(gt_binary_inverse).sum()
    fn = pred_binary_inverse.mul(gt_binary).sum()

    if tp.item() == 0:

        tp = torch.Tensor([1]).cuda()
    
    mae = torch.mean(torch.abs(pred - target))
    sensitivity = tp / (tp + fn + 1e-7)    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
       # Specificity or true negative rate
    Specificity = tn / (tn + fp)

    # Precision or positive predictive value
    Precision = tp / (tp + fp)    
    return {
        'dice': dice.item(),
        'mae': mae.item(),
        'accuracy': accuracy.item(),
        'sensitivity': sensitivity.item(),
        'iou':iou,
        'Specificity':Specificity.item(),
        'Precision':Precision.item()
    }

def validate_model(model, val_loader, device):
    model.eval()
    total_metrics = {
        'dice': 0,
        'mae': 0,
        'accuracy': 0,
        'sensitivity': 0,
        'iou':0,
        'Specificity':0,
        'Precision':0
    }
    num_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            outputs = outputs.squeeze(1)  # Remove channel dimension if necessary
            
            batch_metrics = calculate_metrics(outputs, masks)
            
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key] * images.size(0)
            
            num_samples += images.size(0)
    
    # Calculate average metrics
    avg_metrics = {key: value / num_samples for key, value in total_metrics.items()}
    
    return avg_metrics

# Usage example (you can integrate this into your training loop)
print("Calculating Validation Metrics:")
val_metrics = validate_model(model, test_loader, device)
print("Validation Metrics:")
for key, value in val_metrics.items():
    print(f"{key.capitalize()}: {value:.4f}")

