# Residual Conv-Deconv Grid Network for Semantic Segmentation

This work was published at the British Machine Vision Conference (BMVC) 2017.

The paper is available at : https://arxiv.org/abs/1707.07958

The pretrained model provided is the one used for the paper's evaluation. 

The training code is a refactored version of the one that we used for the paper, and has not yet been tested extensively, so feel free to open an issue if you find any problem.

## Video results

A video of our results on the Cityscapes datasets demo videos is avalaible there : https://youtu.be/jQWpbfj5zsE

## Dataset structure

The code is made to train a GridNet with the Cityscapes dataset.
If you want to train a new model you need to download the dataset (https://www.cityscapes-dataset.com/).

Our code use the environment variable CITYSCAPES_DATASET pointing to the root folder of the dataset.

If you want to evaluate the pretrained model you don't need the dataset.


## Use a pretrained model

You can download a pretrained model at : https://storage.googleapis.com/windy-marker-136923.appspot.com/SHARE/GridNet.t7

Download the pretrained model and put it in the folder pretrained.

```bash
MODEL="pretrained/GridNet.t7" #Pretrained model
FOLDER="$CITYSCAPES_DATASET/leftImg8bit/demoVideo/stuttgart_02/" #Folder containing the images to evaluate

th evaluation.lua -trainLabel -sizeX 400 -sizeY 400 -stepX 300 -stepY 300 -folder $FOLDER -model  $MODEL -rgb -save Test 
```

## Train a model from scratch

You can train a GridNet from scratch using the script train.lua

```bash
th train.lua -extraRatio 0 -scaleMin 1 -scaleMax 2.5 -sizeX 400 -sizeY 400 -hflip -model GridNet -batchSize 4 -nbIterationTrain 750 -nbIterationValid 125
```

## Scripts

Some scripts are given in the folder scripts. 

You can plot the current training evolution using the script plot.sh.
You need to specified which accuracy you want to plot (pixels, class or iou accuracy).
You can plot several accuracy at the same time.

```bash
`./scripts/plot.sh pixels class iou folder_where_the_logs_are
```

## Citation

If you use this code or these models in your research, please cite:

```
@inproceedings{fourure2017gridnet,
  title={Residual Conv-Deconv Grid Network for Semantic Segmentation},
  author={Fourure, Damien and Emonet, R{\'e}mi and Fromont, Elisa and Muselet, Damien and Neverova, Natalia and Tr{\'e}meau, Alain and Wolf, Christian},
  booktitle={Proceedings of the British Machine Vision Conference, 2017},
  year={2017}
}
```

# License

This code is only for academic purpose. For commercial purpose, please contact us.

# Acknowledgement

Authors acknowledge the support from the ANR project SoLStiCe (ANR-13-BS02-0002-01).
We also want to thank NVidia for providing two Titan X GPU.
