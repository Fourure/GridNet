# Residual Conv-Deconv Grid Network for Semantic Segmentation

Paper available at : https://arxiv.org/abs/1707.07958

The pretrained model provided (into the folder pretrained) is the one used for the paper's evaluation. 
The training code is a refactored version of the one that we used for the paper, and has not yet been tested extensively, so feel free to open an issue if you find any problem.

# Video results

A video of our results on the Cityscapes datasets demo videos is avalaible there : https://youtu.be/jQWpbfj5zsE

# Dataset structure

You need to download the Cityscapes dataset.

# Use a pretrained model

You can download a pretrained model at : https://storage.googleapis.com/windy-marker-136923.appspot.com/SHARE/GridNet.t7
Download the pretrained model and put it in the folder pretrained

```bash
MODEL="pretrained/GridNet.t7" #Pretrained model
FOLDER="$CITYSCAPES_DATASET/leftImg8bit/demoVideo/stuttgart_02/" #Folder containing the images to evaluate

th evaluation.lua -trainLabel -sizeX 400 -sizeY 400 -stepX 300 -stepY 300 -folder $FOLDER -model  $MODEL -rgb -save Test 
```

# Train a model from scratch

```bash
th train.lua -extraRatio 0 -scaleMin 1 -scaleMax 2.5 -sizeX 400 -sizeY 400 -hflip -model GridNet -batchSize 4 -nbIterationTrain 750 -nbIterationValid 125
```

# Citation

If you use this code or these models in your research, please cite:

```
@article{fourure2017residual,
  title={Residual Conv-Deconv Grid Network for Semantic Segmentation},
  author={Fourure, Damien and Emonet, R{\'e}mi and Fromont, Elisa and Muselet, Damien and Tremeau, Alain and Wolf, Christian},
  journal={arXiv preprint arXiv:1707.07958},
  year={2017}
}
```

# License

This code is only for academic purpose. For commercial purpose, please contact us.

# Acknowledgement

Authors acknowledge the support from the ANR project SoLStiCe (ANR-13-BS02-0002-01).
We also want to thank NVidia for providing two Titan X GPU.
