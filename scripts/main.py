import torch
import torchvision
import numpy as np
import csv
import time
import utils
import os
import io
import tarfile as tf
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from typing import Tuple, List, Dict
from collections import OrderedDict
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.roi_heads import fastrcnn_loss


###############################################################################
# DATASET
###############################################################################
class SpacecraftImagesDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        Object for handling the spacecraft images dataset
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

        Args:
            annotations_file (str): Path to data labels
            img_dir (str): Path to images
            transform (lambda): Transform to be applied to images
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels
        # self.img_labels = self.img_labels.head(1579)
        self.img_dir = img_dir
        self.pil_to_tensor = ToTensor()
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get the image name and bbox associated with the given idx
        img_name = self.img_labels.iloc[idx, 0]
        bbox = self.img_labels[self.img_labels.image_id == img_name].values[:,1:].astype("float")

        # Get which tar file to grab image from hello
        tar_path = os.path.join(self.img_dir, img_name[0] + ".tar")
        
        # Grab image from tar file
        with tf.open(tar_path, "r") as src_tar:
            member = src_tar.getmember("images/" + img_name + ".png")
            image = src_tar.extractfile(member)
            image = self.pil_to_tensor(Image.open(io.BytesIO(image.read())))

        # Apply transform if input
        if self.transform:
            image = self.transform(image)

        # Define labels in the image
        labels = torch.ones((bbox.shape[0]), dtype=torch.int64)

        target = {}
        target["boxes"] = torch.tensor(bbox, dtype=torch.float32)
        target["labels"] = labels

        return image, target
    
###############################################################################
# CUSTOM VALIDATION LOSS FUNCTION
###############################################################################
def eval_forward(model, images, targets):
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            It returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """
    model.eval()

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    model.rpn.training=True
    #model.roi_heads.training=True


    #####proposals, proposal_losses = model.rpn(images, features, targets)
    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
    model.rpn.training=False
    model.roi_heads.training=False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections


# Select GPU
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Get current time for file timestamps
filetime = time.strftime("%Y_%m_%d_%H_%M_%S")

# Setup directories for output storage
data_dir = "../outputs/" + filetime
inference_dir = data_dir + "/inference"
save_state_dir = data_dir + "/saved_states"

Path("../outputs").mkdir(parents=True, exist_ok=True)
Path(data_dir).mkdir(parents=True, exist_ok=True)
Path(inference_dir).mkdir(parents=True, exist_ok=True)
Path(save_state_dir).mkdir(parents=True, exist_ok=True)

# Data file paths
label_path = "../compressed_data/images/train_labels.csv"
img_dir = "../compressed_data/images"

# Load in the full dataset object
sc_dataset = SpacecraftImagesDataset(label_path, img_dir)

# Take a subset of the dataset to save time
sub_idxs = list(range(0, 25800))
sub_sc_dataset = torch.utils.data.Subset(sc_dataset, sub_idxs)

# Seed the generator
generator = torch.Generator().manual_seed(371)

# Make training, testing, and validation datasets
# 60% training, 20% testing, 20% validation
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(sub_sc_dataset, [0.6, 0.2, 0.2], generator=generator)

###############################################################################
# DATALOADERS
###############################################################################
def collate_fn(data):
    return data

sc_dataloader = DataLoader(sc_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=4,
                            collate_fn=collate_fn,
                            pin_memory=(True if torch.cuda.is_available() else False))

# Create the training, testing, and validation dataloader objects
train_dataloader = DataLoader(train_dataset, 
                              batch_size=16, 
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_fn,
                              pin_memory=(True if torch.cuda.is_available() else False))

test_dataloader = DataLoader(test_dataset, 
                             batch_size=16, 
                             shuffle=False, 
                             num_workers=4,
                             collate_fn=collate_fn,
                             pin_memory=(True if torch.cuda.is_available() else False))

val_dataloader = DataLoader(val_dataset, 
                            batch_size=16, 
                            shuffle=False, 
                            num_workers=4,
                            collate_fn=collate_fn,
                            pin_memory=(True if torch.cuda.is_available() else False))

###############################################################################
# MODEL 
###############################################################################
# Load in pretrained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# Specify the number of classes being used
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained head with new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

###############################################################################
# DEFINE LOSS AND OPTIMIZER
###############################################################################
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

###############################################################################
# TRAINING LOOP
###############################################################################
# Max number of epochs to train for
num_epochs = 20

# Move model to the right device
# model = nn.DataParallel(model)
model.to(device)

train_losses_list = []
val_losses_list = []

loss_csv_filename = data_dir + "/lossdata_" + filetime + ".csv"
loss_csv_fields = ["epoch", "save", "train_loss", "val_loss"]

pbar_total = (num_epochs * len(train_dataloader)) + (num_epochs * len(val_dataloader))

with open(loss_csv_filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(loss_csv_fields)
    with tqdm(total=pbar_total, desc="Training ") as pbar:
        for epoch in range(num_epochs):
            pbar.set_description("Training   ")
            model.train()
            for data in train_dataloader:
                images = [d.to(device) for d, t in data]
                targets = [{k: v.to(device) for (k,v) in t.items()} for d, t in data]

                # Calculate the model loss
                train_loss_dict = model(images, targets)
                train_losses = sum(loss for loss in train_loss_dict.values())
                train_losses_list.append(train_losses.item)
                
                # Backpropagation
                optimizer.zero_grad()
                train_losses.backward()
                optimizer.step()
                
                pbar.update(1)

            pbar.set_description("Validation ")
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0
                for data in val_dataloader:
                    images = [d.to(device) for d, t in data]
                    targets = [{k: v.to(device) for (k,v) in t.items()} for d, t in data]
                    
                    # Calculate the model loss
                    val_loss_dict = eval_forward(model, images, targets)
                    val_losses = sum(loss for loss in val_loss_dict[0].values())
                    val_losses_list.append(val_losses.item())

                    pbar.update(1)

            csvwriter.writerow([epoch, (epoch), train_losses.item(), val_losses.item()])

            pbar.write(f'Epoch {epoch}, Train Loss: {train_losses.item()}, Validation Loss: {val_losses.item()}, Save State: {epoch}')
            torch.save(model.state_dict(), save_state_dir + "/model_weights_" + filetime + f"_s{epoch}.pth")

###############################################################################
# INFERENCE
###############################################################################
# CSV file to store the highest confidence bounding boxes
actual_output_file = inference_dir + "/actual_bboxes_" + filetime + ".csv"

idx = 0

# Open a CSV file to write the ground truth values
with open(actual_output_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(["image_id", "xmin", "ymin", "xmax", "ymax"])

    # Loop through the dataset
    with tqdm(total=(len(test_dataloader))) as pbar:
        for data in test_dataloader:
            for target in data:
                bbox_coords = target[1]['boxes'].numpy().flatten()
                writer.writerow([idx, bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]])
                idx += 1
            pbar.update(1)

# CSV file to store the highest confidence bounding boxes
predicted_output_file = inference_dir + "/predicted_bboxes_" + filetime + ".csv"

# Ensure the model is on the appropriate device and in evaluation mode
model.eval()
model.cpu()

idx = 0

# Open a CSV file to write the ground truth values
with open(predicted_output_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(["image_id", "xmin", "ymin", "xmax", "ymax"])

    # Loop through the dataset
    with tqdm(total=(len(test_dataloader))) as pbar:
        with torch.no_grad():
            for data in test_dataloader:
                images = [d.cpu() for d, t in data]
                targets = [{k: v.cpu() for (k,v) in t.items()} for d, t in data]

                val_dict = model(images, targets)

                for d in val_dict:
                    boxes = d['boxes']
                    scores = d['scores']

                    # Use non-max suppression to get the highest confidence bounding box
                    highest_conf_bbox = torchvision.ops.nms(boxes, scores, 0.5)
                    rows = [[idx, *torch.round(boxes[row_idx]).tolist()] for row_idx in highest_conf_bbox]

                    # Write the index and bbox coords to csv file
                    writer.writerows(rows)
                    idx += 1

                pbar.update(1)
