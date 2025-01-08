# Training and Testing Results for Model 1

## Model Architecture for Model 1
- Input Block: 1 → 8 channels (3x3 conv)
- Convolution Block 1: 8 → 16 → 24 channels (3x3 conv)
- Transition Block: 24 → 8 channels with MaxPooling
- Convolution Block 2: 8 → 16 → 24 channels (3x3 conv)
- Output Block: 24 → 10 → 10 channels (1x1 and 7x7 conv)

## Training Parameters for Model 1
- Optimizer: SGD with momentum (0.9)
- Learning Rate: 0.01
- Batch Size: 128 (GPU) / 64 (CPU)
- Epochs: 15
- Dropout Rate: 0.1

## Receptive Field Analysis for Model 1
```
Layer           RF       n_in     n_out    j_in     j_out    r_in     r_out   
-------------------------------------------------------------------------------
Input           1x1      28       28       1        1        0        0       
Conv1 (3x3)     3x3      28       26       1        1        0        1       
Conv2 (3x3)     5x5      26       24       1        1        1        2       
Conv3 (3x3)     7x7      24       22       1        1        2        3       
MaxPool         14x14    22       11       1        2        3        6       
Conv4 (1x1)     14x14    11       11       2        2        6        6       
Conv5 (3x3)     18x18    11       9        2        2        6        8       
Conv6 (3x3)     22x22    9        7        2        2        8        10      
Conv7 (1x1)     22x22    7        7        2        2        10       10      
Conv8 (7x7)     28x28    7        1        2        14       10       16
```

## Model Parameters for Model 1
```
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
           Dropout-3            [-1, 8, 26, 26]               0
            Conv2d-4           [-1, 16, 24, 24]           1,152
              ReLU-5           [-1, 16, 24, 24]               0
           Dropout-6           [-1, 16, 24, 24]               0
            Conv2d-7           [-1, 24, 22, 22]           3,456
              ReLU-8           [-1, 24, 22, 22]               0
           Dropout-9           [-1, 24, 22, 22]               0
        MaxPool2d-10           [-1, 24, 11, 11]               0
           Conv2d-11            [-1, 8, 11, 11]             192
           Conv2d-12             [-1, 16, 9, 9]           1,152
             ReLU-13             [-1, 16, 9, 9]               0
          Dropout-14             [-1, 16, 9, 9]               0
           Conv2d-15             [-1, 24, 7, 7]           3,456
             ReLU-16             [-1, 24, 7, 7]               0
          Dropout-17             [-1, 24, 7, 7]               0
           Conv2d-18             [-1, 10, 7, 7]             240
           Conv2d-19             [-1, 10, 1, 1]           4,900
================================================================
Total params: 14,620
Trainable params: 14,620
Non-trainable params: 0
```

## Accuracy Logs for Model 1
```
2025-01-08 13:00:10 | INFO | EPOCH: 0
2025-01-08 13:00:15 | INFO | Loss=0.282642513513565 Batch_id=468 Accuracy=65.90
2025-01-08 13:00:16 | INFO | Test set: Average loss: 0.1475, Accuracy: 9568/10000 (95.68%)
2025-01-08 13:00:16 | INFO | EPOCH: 1
2025-01-08 13:00:22 | INFO | Loss=0.067930340766907 Batch_id=468 Accuracy=95.89
2025-01-08 13:00:23 | INFO | Test set: Average loss: 0.0735, Accuracy: 9765/10000 (97.65%)
2025-01-08 13:00:23 | INFO | EPOCH: 2
2025-01-08 13:00:28 | INFO | Loss=0.016118794679642 Batch_id=468 Accuracy=97.29
2025-01-08 13:00:29 | INFO | Test set: Average loss: 0.0535, Accuracy: 9829/10000 (98.29%)
2025-01-08 13:00:29 | INFO | EPOCH: 3
2025-01-08 13:00:35 | INFO | Loss=0.025231121107936 Batch_id=468 Accuracy=97.76
2025-01-08 13:00:36 | INFO | Test set: Average loss: 0.0511, Accuracy: 9833/10000 (98.33%)
2025-01-08 13:00:36 | INFO | EPOCH: 4
2025-01-08 13:00:41 | INFO | Loss=0.138481006026268 Batch_id=468 Accuracy=98.04
2025-01-08 13:00:42 | INFO | Test set: Average loss: 0.0461, Accuracy: 9844/10000 (98.44%)
2025-01-08 13:00:42 | INFO | EPOCH: 5
2025-01-08 13:00:48 | INFO | Loss=0.010701477527618 Batch_id=468 Accuracy=98.25
2025-01-08 13:00:49 | INFO | Test set: Average loss: 0.0428, Accuracy: 9868/10000 (98.68%)
2025-01-08 13:00:49 | INFO | EPOCH: 6
2025-01-08 13:00:54 | INFO | Loss=0.002213725587353 Batch_id=468 Accuracy=98.47
2025-01-08 13:00:55 | INFO | Test set: Average loss: 0.0359, Accuracy: 9868/10000 (98.68%)
2025-01-08 13:00:55 | INFO | EPOCH: 7
2025-01-08 13:01:01 | INFO | Loss=0.138315185904503 Batch_id=468 Accuracy=98.53
2025-01-08 13:01:02 | INFO | Test set: Average loss: 0.0355, Accuracy: 9876/10000 (98.76%)
2025-01-08 13:01:02 | INFO | EPOCH: 8
2025-01-08 13:01:07 | INFO | Loss=0.053662225604057 Batch_id=468 Accuracy=98.57
2025-01-08 13:01:08 | INFO | Test set: Average loss: 0.0333, Accuracy: 9883/10000 (98.83%)
2025-01-08 13:01:08 | INFO | EPOCH: 9
2025-01-08 13:01:14 | INFO | Loss=0.037520784884691 Batch_id=468 Accuracy=98.69
2025-01-08 13:01:15 | INFO | Test set: Average loss: 0.0347, Accuracy: 9880/10000 (98.80%)
2025-01-08 13:01:15 | INFO | EPOCH: 10
2025-01-08 13:01:20 | INFO | Loss=0.019322413951159 Batch_id=468 Accuracy=98.70
2025-01-08 13:01:21 | INFO | Test set: Average loss: 0.0334, Accuracy: 9881/10000 (98.81%)
2025-01-08 13:01:21 | INFO | EPOCH: 11
2025-01-08 13:01:27 | INFO | Loss=0.011548400856555 Batch_id=468 Accuracy=98.84
2025-01-08 13:01:28 | INFO | Test set: Average loss: 0.0304, Accuracy: 9887/10000 (98.87%)
2025-01-08 13:01:28 | INFO | EPOCH: 12
2025-01-08 13:01:33 | INFO | Loss=0.017873024567962 Batch_id=468 Accuracy=98.81
2025-01-08 13:01:34 | INFO | Test set: Average loss: 0.0319, Accuracy: 9888/10000 (98.88%)
2025-01-08 13:01:34 | INFO | EPOCH: 13
2025-01-08 13:01:40 | INFO | Loss=0.016239201650023 Batch_id=468 Accuracy=98.89
2025-01-08 13:01:41 | INFO | Test set: Average loss: 0.0276, Accuracy: 9897/10000 (98.97%)
2025-01-08 13:01:41 | INFO | EPOCH: 14
2025-01-08 13:01:46 | INFO | Loss=0.027947871014476 Batch_id=468 Accuracy=98.86
2025-01-08 13:01:47 | INFO | Test set: Average loss: 0.0293, Accuracy: 9898/10000 (98.98%)
```


# Training and Testing Results for Model 2

## Model Architecture for Model 2
- Input Block: 1 → 8 channels (3x3 conv)
- Convolution Block 1: 8 → 16 → 24 channels (3x3 conv)
- Transition Block: 24 → 8 channels with MaxPooling
- Convolution Block 2: 8 → 16 → 24 channels (3x3 conv)
- Output Block: 24 → 10 → 10 channels (1x1 and 7x7 conv)

## Training Parameters
- Optimizer: SGD with momentum (0.9)
- Learning Rate: 0.01
- Batch Size: 128 (GPU) / 64 (CPU)
- Epochs: 15
- Dropout Rate: 0.1

## Receptive Field Analysis for Model 2
```
Layer           RF       n_in     n_out    j_in     j_out    r_in     r_out   
-------------------------------------------------------------------------------
Input           1x1      28       28       1        1        0        0       
Conv1 (3x3)     3x3      28       26       1        1        0        1       
Conv2 (3x3)     5x5      26       24       1        1        1        2       
Conv3 (3x3)     7x7      24       22       1        1        2        3       
MaxPool         14x14    22       11       1        2        3        6       
Conv4 (1x1)     14x14    11       11       2        2        6        6       
Conv5 (3x3)     18x18    11       9        2        2        6        8       
Conv6 (3x3)     22x22    9        7        2        2        8        10      
Conv7 (1x1)     22x22    7        7        2        2        10       10      
Conv8 (7x7)     28x28    7        1        2        14       10       16
```

## Model Parameters for Model 2
```
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           1,152
       BatchNorm2d-6           [-1, 16, 24, 24]              32
           Dropout-7           [-1, 16, 24, 24]               0
            Conv2d-8           [-1, 24, 22, 22]           3,456
              ReLU-9           [-1, 24, 22, 22]               0
      BatchNorm2d-10           [-1, 24, 22, 22]              48
          Dropout-11           [-1, 24, 22, 22]               0
        MaxPool2d-12           [-1, 24, 11, 11]               0
           Conv2d-13            [-1, 8, 11, 11]             192
           Conv2d-14             [-1, 16, 9, 9]           1,152
             ReLU-15             [-1, 16, 9, 9]               0
      BatchNorm2d-16             [-1, 16, 9, 9]              32
          Dropout-17             [-1, 16, 9, 9]               0
           Conv2d-18             [-1, 24, 7, 7]           3,456
             ReLU-19             [-1, 24, 7, 7]               0
      BatchNorm2d-20             [-1, 24, 7, 7]              48
          Dropout-21             [-1, 24, 7, 7]               0
           Conv2d-22             [-1, 10, 7, 7]             240
           Conv2d-23             [-1, 10, 1, 1]           4,900
================================================================
Total params: 14,796
Trainable params: 14,796
Non-trainable params: 0
```

## Accuracy Logs for Model 2
```
2025-01-08 12:45:39 | INFO | EPOCH: 0
2025-01-08 12:45:47 | INFO | Loss=0.089236907660961 Batch_id=468 Accuracy=94.31
2025-01-08 12:45:48 | INFO | Test set: Average loss: 0.0550, Accuracy: 9829/10000 (98.29%)
2025-01-08 12:45:48 | INFO | EPOCH: 1
2025-01-08 12:45:54 | INFO | Loss=0.041903179138899 Batch_id=468 Accuracy=98.19
2025-01-08 12:45:55 | INFO | Test set: Average loss: 0.0415, Accuracy: 9868/10000 (98.68%)
2025-01-08 12:45:55 | INFO | EPOCH: 2
2025-01-08 12:46:00 | INFO | Loss=0.029992125928402 Batch_id=468 Accuracy=98.64
2025-01-08 12:46:01 | INFO | Test set: Average loss: 0.0310, Accuracy: 9909/10000 (99.09%)
2025-01-08 12:46:01 | INFO | EPOCH: 3
2025-01-08 12:46:07 | INFO | Loss=0.006373236421496 Batch_id=468 Accuracy=98.77
2025-01-08 12:46:08 | INFO | Test set: Average loss: 0.0344, Accuracy: 9881/10000 (98.81%)
2025-01-08 12:46:08 | INFO | EPOCH: 4
2025-01-08 12:46:13 | INFO | Loss=0.037650588899851 Batch_id=468 Accuracy=98.85
2025-01-08 12:46:14 | INFO | Test set: Average loss: 0.0246, Accuracy: 9919/10000 (99.19%)
2025-01-08 12:46:14 | INFO | EPOCH: 5
2025-01-08 12:46:20 | INFO | Loss=0.009082222357392 Batch_id=468 Accuracy=98.95
2025-01-08 12:46:20 | INFO | Test set: Average loss: 0.0280, Accuracy: 9905/10000 (99.05%)
2025-01-08 12:46:20 | INFO | EPOCH: 6
2025-01-08 12:46:26 | INFO | Loss=0.005864668171853 Batch_id=468 Accuracy=99.03
2025-01-08 12:46:27 | INFO | Test set: Average loss: 0.0295, Accuracy: 9914/10000 (99.14%)
2025-01-08 12:46:27 | INFO | EPOCH: 7
2025-01-08 12:46:32 | INFO | Loss=0.080738790333271 Batch_id=468 Accuracy=99.14
2025-01-08 12:46:33 | INFO | Test set: Average loss: 0.0253, Accuracy: 9924/10000 (99.24%)
2025-01-08 12:46:33 | INFO | EPOCH: 8
2025-01-08 12:46:39 | INFO | Loss=0.006130995228887 Batch_id=468 Accuracy=99.17
2025-01-08 12:46:40 | INFO | Test set: Average loss: 0.0274, Accuracy: 9902/10000 (99.02%)
2025-01-08 12:46:40 | INFO | EPOCH: 9
2025-01-08 12:46:45 | INFO | Loss=0.007717892993242 Batch_id=468 Accuracy=99.16
2025-01-08 12:46:46 | INFO | Test set: Average loss: 0.0246, Accuracy: 9923/10000 (99.23%)
2025-01-08 12:46:46 | INFO | EPOCH: 10
2025-01-08 12:46:52 | INFO | Loss=0.00892304815352 Batch_id=468 Accuracy=99.23
2025-01-08 12:46:53 | INFO | Test set: Average loss: 0.0308, Accuracy: 9908/10000 (99.08%)
2025-01-08 12:46:53 | INFO | EPOCH: 11
2025-01-08 12:46:58 | INFO | Loss=0.003149656578898 Batch_id=468 Accuracy=99.26
2025-01-08 12:46:59 | INFO | Test set: Average loss: 0.0247, Accuracy: 9923/10000 (99.23%)
2025-01-08 12:46:59 | INFO | EPOCH: 12
2025-01-08 12:47:05 | INFO | Loss=0.007260935381055 Batch_id=468 Accuracy=99.35
2025-01-08 12:47:06 | INFO | Test set: Average loss: 0.0239, Accuracy: 9922/10000 (99.22%)
2025-01-08 12:47:06 | INFO | EPOCH: 13
2025-01-08 12:47:11 | INFO | Loss=0.005321543198079 Batch_id=468 Accuracy=99.28
2025-01-08 12:47:12 | INFO | Test set: Average loss: 0.0234, Accuracy: 9929/10000 (99.29%)
2025-01-08 12:47:12 | INFO | EPOCH: 14
2025-01-08 12:47:18 | INFO | Loss=0.014144137501717 Batch_id=468 Accuracy=99.33
2025-01-08 12:47:19 | INFO | Test set: Average loss: 0.0226, Accuracy: 9929/10000 (99.29%)
```

# Training and Testing Results for Model 3

## Model Architecture for Model 3
- Input Block: 1 → 8 channels (3x3 conv)
- Convolution Block 1: 8 → 8 → 16 channels (3x3 conv)
- Transition Block: 16 → 8 channels with MaxPooling
- Convolution Block 2: 8 → 8 → 16 channels (3x3 conv)
- Output Block: 16 → 10 → 10 channels (1x1 and 7x7 conv)

## Model Parameters for Model 3
```
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5            [-1, 8, 24, 24]             576
       BatchNorm2d-6            [-1, 8, 24, 24]              16
           Dropout-7            [-1, 8, 24, 24]               0
            Conv2d-8           [-1, 16, 22, 22]           1,152
              ReLU-9           [-1, 16, 22, 22]               0
      BatchNorm2d-10           [-1, 16, 22, 22]              32
          Dropout-11           [-1, 16, 22, 22]               0
        MaxPool2d-12           [-1, 16, 11, 11]               0
           Conv2d-13            [-1, 8, 11, 11]             128
           Conv2d-14              [-1, 8, 9, 9]             576
             ReLU-15              [-1, 8, 9, 9]               0
      BatchNorm2d-16              [-1, 8, 9, 9]              16
          Dropout-17              [-1, 8, 9, 9]               0
           Conv2d-18             [-1, 16, 7, 7]           1,152
             ReLU-19             [-1, 16, 7, 7]               0
      BatchNorm2d-20             [-1, 16, 7, 7]              32
          Dropout-21             [-1, 16, 7, 7]               0
           Conv2d-22             [-1, 10, 7, 7]             160
        AvgPool2d-23             [-1, 10, 1, 1]               0
================================================================
Total params: 3,928
Trainable params: 3,928
Non-trainable params: 0
```

## Accuracy Logs for Model 3
Accuracy has dropped to 98.51 in best case. Not worth exploring further.

# Training and Testing Results for Model 4

## Model Architecture for Model 4
- Input Block: 1 → 8 channels (3x3 conv)
- Convolution Block 1: 8 → 16 → 16 channels (3x3 conv)
- Transition Block: 16 → 8 channels with MaxPooling
- Convolution Block 2: 8 → 16 → 16 channels (3x3 conv)
- Output Block: 16 → 10 → 10 channels (1x1 and 7x7 conv)

## Model Parameters for Model 4
```
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           1,152
       BatchNorm2d-6           [-1, 16, 24, 24]              32
           Dropout-7           [-1, 16, 24, 24]               0
            Conv2d-8           [-1, 16, 22, 22]           2,304
              ReLU-9           [-1, 16, 22, 22]               0
      BatchNorm2d-10           [-1, 16, 22, 22]              32
          Dropout-11           [-1, 16, 22, 22]               0
        MaxPool2d-12           [-1, 16, 11, 11]               0
           Conv2d-13            [-1, 8, 11, 11]             128
           Conv2d-14             [-1, 16, 9, 9]           1,152
             ReLU-15             [-1, 16, 9, 9]               0
      BatchNorm2d-16             [-1, 16, 9, 9]              32
          Dropout-17             [-1, 16, 9, 9]               0
           Conv2d-18             [-1, 16, 7, 7]           2,304
             ReLU-19             [-1, 16, 7, 7]               0
      BatchNorm2d-20             [-1, 16, 7, 7]              32
          Dropout-21             [-1, 16, 7, 7]               0
           Conv2d-22             [-1, 10, 7, 7]             160
        AvgPool2d-23             [-1, 10, 1, 1]               0
================================================================
Total params: 7,416
Trainable params: 7,416
Non-trainable params: 0
```

## Accuracy Logs for Model 4
```
2025-01-08 13:34:56 | INFO | EPOCH: 0
2025-01-08 13:35:02 | INFO | Loss=0.21636663377285 Batch_id=468 Accuracy=80.64
2025-01-08 13:35:03 | INFO | Test set: Average loss: 0.2643, Accuracy: 9258/10000 (92.58%)
2025-01-08 13:35:03 | INFO | EPOCH: 1
2025-01-08 13:35:08 | INFO | Loss=0.13411471247673 Batch_id=468 Accuracy=96.65
2025-01-08 13:35:09 | INFO | Test set: Average loss: 0.0996, Accuracy: 9743/10000 (97.43%)
2025-01-08 13:35:09 | INFO | EPOCH: 2
2025-01-08 13:35:15 | INFO | Loss=0.133614853024483 Batch_id=468 Accuracy=97.50
2025-01-08 13:35:16 | INFO | Test set: Average loss: 0.0773, Accuracy: 9801/10000 (98.01%)
2025-01-08 13:35:16 | INFO | EPOCH: 3
2025-01-08 13:35:21 | INFO | Loss=0.048495646566153 Batch_id=468 Accuracy=97.91
2025-01-08 13:35:22 | INFO | Test set: Average loss: 0.0646, Accuracy: 9814/10000 (98.14%)
2025-01-08 13:35:22 | INFO | EPOCH: 4
2025-01-08 13:35:27 | INFO | Loss=0.080205358564854 Batch_id=468 Accuracy=98.15
2025-01-08 13:35:28 | INFO | Test set: Average loss: 0.0526, Accuracy: 9839/10000 (98.39%)
2025-01-08 13:35:28 | INFO | EPOCH: 5
2025-01-08 13:35:34 | INFO | Loss=0.023551568388939 Batch_id=468 Accuracy=98.33
2025-01-08 13:35:35 | INFO | Test set: Average loss: 0.0487, Accuracy: 9845/10000 (98.45%)
2025-01-08 13:35:35 | INFO | EPOCH: 6
2025-01-08 13:35:40 | INFO | Loss=0.06669194996357 Batch_id=468 Accuracy=98.45
2025-01-08 13:35:41 | INFO | Test set: Average loss: 0.0443, Accuracy: 9871/10000 (98.71%)
2025-01-08 13:35:41 | INFO | EPOCH: 7
2025-01-08 13:35:47 | INFO | Loss=0.085388988256454 Batch_id=468 Accuracy=98.58
2025-01-08 13:35:48 | INFO | Test set: Average loss: 0.0433, Accuracy: 9872/10000 (98.72%)
2025-01-08 13:35:48 | INFO | EPOCH: 8
2025-01-08 13:35:53 | INFO | Loss=0.102255403995514 Batch_id=468 Accuracy=98.56
2025-01-08 13:35:54 | INFO | Test set: Average loss: 0.0340, Accuracy: 9904/10000 (99.04%)
2025-01-08 13:35:54 | INFO | EPOCH: 9
2025-01-08 13:35:59 | INFO | Loss=0.027389526367188 Batch_id=468 Accuracy=98.63
2025-01-08 13:36:00 | INFO | Test set: Average loss: 0.0328, Accuracy: 9898/10000 (98.98%)
2025-01-08 13:36:00 | INFO | EPOCH: 10
2025-01-08 13:36:06 | INFO | Loss=0.023722825571895 Batch_id=468 Accuracy=98.70
2025-01-08 13:36:07 | INFO | Test set: Average loss: 0.0327, Accuracy: 9903/10000 (99.03%)
2025-01-08 13:36:07 | INFO | EPOCH: 11
2025-01-08 13:36:12 | INFO | Loss=0.052927505224943 Batch_id=468 Accuracy=98.77
2025-01-08 13:36:13 | INFO | Test set: Average loss: 0.0358, Accuracy: 9892/10000 (98.92%)
2025-01-08 13:36:13 | INFO | EPOCH: 12
2025-01-08 13:36:19 | INFO | Loss=0.025599950924516 Batch_id=468 Accuracy=98.82
2025-01-08 13:36:20 | INFO | Test set: Average loss: 0.0339, Accuracy: 9885/10000 (98.85%)
2025-01-08 13:36:20 | INFO | EPOCH: 13
2025-01-08 13:36:25 | INFO | Loss=0.021234704181552 Batch_id=468 Accuracy=98.88
2025-01-08 13:36:26 | INFO | Test set: Average loss: 0.0306, Accuracy: 9909/10000 (99.09%)
2025-01-08 13:36:26 | INFO | EPOCH: 14
2025-01-08 13:36:32 | INFO | Loss=0.040463428944349 Batch_id=468 Accuracy=98.85
2025-01-08 13:36:33 | INFO | Test set: Average loss: 0.0310, Accuracy: 9902/10000 (99.02%)
```

# Conclusions
- Model 2 is better than Model 1, interms of accuracy (99.29% vs 98.98%) and loss (0.0226 vs 0.0293), however it has more parameters (14,796) than Model 1 (14,620). But we need to get number of parameters below 8000 and accuracy above 99.4%.
- Model 4 is better in terms of parameters and not accuracy. We had better accuracy with less parameters in Model 2.
- Reduced accuracy in Model 4 is due to the change in dropout rate from 10% to 15%.
- The best case accurancy in model 4 is 99.13% after making the dropout rate to 7.5%. But we need something better than that. So we will stick to 10% dropout rate, as it has more consistancy in last 2 epochs.
- Model 3 & 4 we have used AvgPooling in output block. We will try to remove AvgPooling and see the changes in accuracy in model 4, but we need something to not increase the number of parameters. So 1. We added AvgPooling in output block then a convolution. and 2. we added image augmentation. 


