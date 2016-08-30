# This repository is based on [A Large-Scale Car Dataset for Fine-Grained Categorization and Verification](http://arxiv.org/abs/1506.08959)
## [The website](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)
## [Github](https://gist.github.com/bogger/b90eb88e31cd745525ae)

### (The dataset is not open. You have to follow their instructions to download it)

## Here is the summary of the paper:
#### Dataset
*	Comprehensive Cars(CompCars)
*	208,826(total images) = 136,727(web-nature, entire car) + 27,618(web-nature, car parts) + 44481(surveillance-nature)
*	web-nature: from forums, websites
*	surveillance-nature: by cameras, front view, with bounding box, model, color
*	car hierarchy, car attributes, viewpoints, car parts
	+	Car hierarchy
		-	make(163), model(1716), year
	+	Car attributes(5)
		-	explicit(discrete values):
			*	\# doors, \# seats, type of car(12)
			*	type of car: MPV, SUV, hatchback, sedan, minibus, fastback, estate, pickup,sports, crossover, convertible, hardtop convertible
		-	implicit(continuous values):
			*	maximum speed, displacement
	+	Viewpoints:
		-	front(F)(18431), rear(R)(13513), side(S)(23551), front-side(FS)(49301),	rear-side(RS)(31150)
		-	18431+13513+23551+49301+31150=135946 ≠ 136727
	+	Car parts:
		-	exterior:
			*	headlight(3705), taillight(3563), fog light(3177), air intake(3407)
		-	interior:
			*	console(3350), steering wheel(3503), dashboard(3478), gear lever(3435)
		-	3705+3563+3177+3407+3350+3503+3478+3435=27618
	+	(The # model in No. per model of Viewpoint and Car part seems not 1716, nearly 1687)

#### Applications
*	fine-grained car classification, attribute prediction, car verification
*	78126 images to 3 subsets
	+	Part-I: 431 models, 30955 images, for car classfication and training of attribute prediction
	+	Part-II: 111 models, 4454 images, for testing of attribute prediction, training of car verification
	+	Part-III: 1145 models, 22236 images, for testing of car verification
	+	30955+20349+4454+22236=77994 ≠ 78126
	+	431+111+1145=1687 ≠ 1716
*	Fine-Grained Classification
	+	Target: car model labels
	+	431 models, different years as same category.
	+	The Entire Car Images:
		-	different viewpoints: F, R, S, FS, RS, All
		-	FS,RS is better. All is surprisingly good: CNN can learn itself across different viewpoints
		-	wrong predictions: same make. Coarse-to-fine(make to model) is possible
		-	web-nature is potential to be transferred to data in other scenario
	+	Car Parts:
		-	taillight is the best
		-	voting strategy

*	Attribute Prediction
	+	explicit(discrete): # door(2,3,4,5), # seat(2,4,5>5), car type(12)
	+	implicit(continuous): maximum speed, displacement
	+	finetune: 
		-	continuous: sum-of-square loss
		-	discrete: logistic loss
	+	“mean guess”: prediction = mean of the training set
	+	“mean difference”: difference of mean
*	Car Verification
	+	Train on Part-II, test on Part-III
	+	easy, medium, hard, each has 10,000 positive pairs, 10,000 negative pairs
		-	easy: same viewpoint
		-	medium: random viewpoints
		-	hard: same car make
	+	Joint Bayesian
	+	CNN+Joint Bayesian > CNN+SVM
	+	Challenges: 
		-	same model, different viewpoints: difficult to obtain correspondences
		-	same make, different models: extremely similar, difficult to distinguish
#### Update Result
*	entire dataset
*	three subsets:
	+	car models: 431, 111, 1145
	+	images: 52083, 11129, 72962 -> 136174 ≠ 136727
	+	AlexNet < Overfeat < GoogLeNet, pretrained on ImageNet
*	Surveillance Data:
	+	281 models
	+	very high












## Experiment Flow:
![](car.jpg)

