# This repository is based on [A Large-Scale Car Dataset for Fine-Grained Categorization and Verification](http://arxiv.org/abs/1506.08959)
## [The website](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)
## [Github Gist](https://gist.github.com/bogger/b90eb88e31cd745525ae)

## (The dataset is not open. You have to follow their instructions to download it)

## See [summary.md](https://github.com/Lorne0/car_classification/blob/master/summary.md) to know the summary of the paper

## Experiment Flow:
![](car.jpg)

## There are 6 experiments as you can see in the picture above:
ex1.	pre-trained ImageNet model -> Web model (the model is provided by the author in the Github Gist)
ex2.	Web model -> Web best model
ex3.	[pre-trained ImageNet model](http://vision.princeton.edu/pvt/GoogLeNet/) -> SV model
ex4.	Web best model -> SV model
ex5.	pre-trained ImageNet model -> Mix(Web+SV) as 431 class
ex6.	pre-trained ImageNet model -> Mix(Web+SV) as 531 class
#### The last two experiments are different because the intersection of Web and SV are only 181 classes, not all 281 classes in SV     
#### So 431 class is for those who have intersection, 531 are their union


