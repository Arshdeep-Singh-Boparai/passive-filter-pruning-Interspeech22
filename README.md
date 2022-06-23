# Passive Pruning 

```
(work in progress)
```
The project aims at identifying unimportant filters in a pre-trained CNN via pruning approach that considers relationship between filters. The project is a part of AI4S project.

A brief description of *.py and other folders/links  is given below,

# Baseline_code.py
The script runs a baseline netowrk or unpruned network (DCASE 2021 task 1a) and gives accuracy (48.58%) and log-loss (1.425) as output.# Fine_tuning.py
This is script used to obtain pruned model from a given set of important filter indexes and performs fine-tuning of the pruned network.


# Importance_calculation.py
 This script  calculates indexes of the important filters in a given intermediate layer of CNN.

# Evaluation_pruned_model.py

Evaluate a pruned model. The Pruned model is in folder 'Pruned_model'.

# Fine-tuning.py

The script is used to fine-tune the pruned network.

# Required libraries


keras, tensorflow, h5py, sklearn, scipy.


# Data downloading link
Download the training, validation dataset (numpy files) and the pre-trained model or weights from the following link:

https://drive.google.com/drive/folders/1qE8La-LgP7GL1wCzPuucHeugkLWHU26a?usp=sharing

The numpy dataset files are the audio features (Log mel-band energies (40 bands) and 500 time frames) as used by the DCASE 2021 task 1a baseline model. 

To download raw audio files, please follow the DCASE 2021 task 1a link: https://dcase.community/challenge2021/task-acoustic-scene-classification#subtask-a  and follow the link: https://github.com/marmoi/dcase2021_task1a_baseline to generate audio features.

```
Please keep the downloaded numpy files (data files) into "~/DCASE2021" folder.
```

# Folder: Pruned_model

The directory contains pruned model obtained after fine-tuning process.


# Folder: Importance_index_layerwise

The directory contains important filter indexes obtained after applying the proposed algorithm (Importance_calculation.py).

# Folder: Scripts to generate Figures

The directory contains various scripts to generate Figures/plots(#2,#3,#5,#6) shown in our paper, submitted to Interspeech2022: https://arxiv.org/pdf/2203.15751.pdf.

# Folder: Quantized_model

The directory contains quantized models obtained after conversion of paramters to float16 or deafult (tflite). The layerwise pruned model converted into tflite, compressed to float16 or default is included. 
```
To generate various paramters from the following Table, please run "Quantized_model.py" script.
```
| Pruned Layer | Accuracy (%) float32 | Accuracy (%) float16 | Accuracy (%) unit8 | size (KB)  float32 | size (KB)  float16 | size (KB)  uint8 |
|:------------:|:--------------------:|:--------------------:|:--------------------:|:------------------:|:------------------:|:----------------:|
|      C1      |         48.31        |         48.28        |         48.42        |         164        |         87         |        50        |
|      C2      |         47.61        |         47.64        |         47.30        |         135        |         72         |        43        |
|      C3      |         47.78        |         47.84        |         47.74        |         143        |         76         |        45        |
|     C1+C2    |         46.93        |         46.93        |         46.69        |         124        |         67         |        40        |
|     C2+C3    |         45.35        |         45.38        |         45.42        |         107        |         58         |        36        |
|   C1+C2+C3   |         45.01        |         44.98        |         44.81        |         96         |         53         |        33        |
