# DNN-Results-on-Negative-Images (Currently only on CNN)
This repository is created after a discussion on a paper which can
be found [here.](https://arxiv.org/pdf/1703.06857.pdf%20-%20Deep%20Neural%20Networks%20Do%20Not%20Recognize%20Negative%20Images)

Thanks to [Fuat Beser](https://www.linkedin.com/in/fuatbeser/) for his presentation on a group called Derin Ogrenme.

# How to use the script

1. Clone the repository and cd into the directory
2. Run ```pip install --editable .``` to install the CLI
3. Run the script using ```trainer --model {architecture-name}```
4. Current architectures are
  * lenet_5
  * mvgg_5
  * mvgg_6
  * mvgg_7
  * mvgg_8
  * mvgg_9
5. You can always use trainer --help command to see the options available
 
##What's done in this repo?
1. Training regular MNIST images with a simple CNN model
2. Converting regular images into negative ones and training the model only on negative images dataset.
3. Combining both regular and negative images and train a model on this combined dataset

##To-Do
1. Improve the CLI. 
2. Simply find a way to fix the issue with the negative images! :)