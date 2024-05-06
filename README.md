# Spacecraft Detection in Images Using Machine Learning
Authors:
- Nick Duggan
- Robert Mancini
- Luke Spinosa

## Running the Code
### Training
All training occurs in the `scripts/main.py` file. To train the model, you must first be inside of the scripts directory
```bash
cd scripts
```
then run the file with the following command:
```bash
python main.py
```
A progress bar will display while training is occuring indicating the current batch being processed. Every epoch the loss data will be printed to the console and the models weights and loss data will be saved to a timestamped folder inside of a directory named `outputs` that will be created automatically in the root directory.

## Network Architecture
```
=========================================================================================================
Layer (type:depth-idx)                                  Output Shape              Param #
=========================================================================================================
FasterRCNN                                              [100, 4]                  --
├─GeneralizedRCNNTransform: 1-1                         [16, 3, 1024, 800]        --
├─BackboneWithFPN: 1-2                                  [16, 256, 16, 13]         --
│    └─IntermediateLayerGetter: 2-1                     [16, 2048, 32, 25]        --
│    │    └─Conv2d: 3-1                                 [16, 64, 512, 400]        (9,408)
│    │    └─FrozenBatchNorm2d: 3-2                      [16, 64, 512, 400]        --
│    │    └─ReLU: 3-3                                   [16, 64, 512, 400]        --
│    │    └─MaxPool2d: 3-4                              [16, 64, 256, 200]        --
│    │    └─Sequential: 3-5                             [16, 256, 256, 200]       (212,992)
│    │    └─Sequential: 3-6                             [16, 512, 128, 100]       1,212,416
│    │    └─Sequential: 3-7                             [16, 1024, 64, 50]        7,077,888
│    │    └─Sequential: 3-8                             [16, 2048, 32, 25]        14,942,208
│    └─FeaturePyramidNetwork: 2-2                       [16, 256, 16, 13]         --
│    │    └─ModuleList: 3-15                            --                        (recursive)
│    │    └─ModuleList: 3-16                            --                        (recursive)
│    │    └─ModuleList: 3-15                            --                        (recursive)
│    │    └─ModuleList: 3-16                            --                        (recursive)
│    │    └─ModuleList: 3-15                            --                        (recursive)
│    │    └─ModuleList: 3-16                            --                        (recursive)
│    │    └─ModuleList: 3-15                            --                        (recursive)
│    │    └─ModuleList: 3-16                            --                        (recursive)
│    │    └─LastLevelMaxPool: 3-17                      [16, 256, 256, 200]       --
├─RegionProposalNetwork: 1-3                            [1000, 4]                 --
│    └─RPNHead: 2-3                                     [16, 3, 256, 200]         --
│    │    └─Sequential: 3-18                            [16, 256, 256, 200]       590,080
│    │    └─Conv2d: 3-19                                [16, 3, 256, 200]         771
│    │    └─Conv2d: 3-20                                [16, 12, 256, 200]        3,084
│    │    └─Sequential: 3-21                            [16, 256, 128, 100]       (recursive)
│    │    └─Conv2d: 3-22                                [16, 3, 128, 100]         (recursive)
│    │    └─Conv2d: 3-23                                [16, 12, 128, 100]        (recursive)
│    │    └─Sequential: 3-24                            [16, 256, 64, 50]         (recursive)
│    │    └─Conv2d: 3-25                                [16, 3, 64, 50]           (recursive)
│    │    └─Conv2d: 3-26                                [16, 12, 64, 50]          (recursive)
│    │    └─Sequential: 3-27                            [16, 256, 32, 25]         (recursive)
│    │    └─Conv2d: 3-28                                [16, 3, 32, 25]           (recursive)
│    │    └─Conv2d: 3-29                                [16, 12, 32, 25]          (recursive)
│    │    └─Sequential: 3-30                            [16, 256, 16, 13]         (recursive)
│    │    └─Conv2d: 3-31                                [16, 3, 16, 13]           (recursive)
│    │    └─Conv2d: 3-32                                [16, 12, 16, 13]          (recursive)
│    └─AnchorGenerator: 2-4                             [204624, 4]               --
├─RoIHeads: 1-4                                         [100, 4]                  --
│    └─MultiScaleRoIAlign: 2-5                          [16000, 256, 7, 7]        --
│    └─TwoMLPHead: 2-6                                  [16000, 1024]             --
│    │    └─Linear: 3-33                                [16000, 1024]             12,846,080
│    │    └─Linear: 3-34                                [16000, 1024]             1,049,600
│    └─FastRCNNPredictor: 2-7                           [16000, 2]                --
│    │    └─Linear: 3-35                                [16000, 2]                2,050
│    │    └─Linear: 3-36                                [16000, 8]                8,200
=========================================================================================================
Total params: 41,299,161
Trainable params: 41,076,761
Non-trainable params: 222,400
Total mult-adds (Units.TERABYTES): 2.68
=========================================================================================================
Input size (MB): 251.66
Forward/backward pass size (MB): 30311.83
Params size (MB): 165.20
Estimated Total Size (MB): 30728.68
=========================================================================================================
```