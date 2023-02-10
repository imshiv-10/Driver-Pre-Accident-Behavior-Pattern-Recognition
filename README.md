# Driver-Pre-Accident-Behavior-Pattern-Recognition


# Abstract

Driving a car is a complex task, and it requires complete attention. Distracted driving is any activity that takes away the driver’s attention from the road. Several studies have identified three main types of distraction: visual distractions (driver’s eyes off the road), manual distractions (driver’s hands off the wheel) and cognitive distractions (driver’s mind off the driving task). The National Highway Traffic Safety Administration (NHTSA) reported that 36,750 people died in motor vehicle crashes in 2018, and 12% of it was due to distracted driving. Texting is the most alarming distraction. Sending or reading a text takes your eyes off the road for 5 seconds. At 55 mph, that’s like driving the length of an entire football field with your eyes closed. Many states now have laws against texting, talking on a cell phone, and other distractions while driving. We believe that computer vision can augment the efforts of the governments to prevent accidents caused by distracted driving. Our algorithm automatically detects the distracted activity of the drivers and alerts them. We envision this type of product being embedded in cars to prevent accidents due to distracted driving. We employed algorithms like Adam, Exception, and ResNet50 to achieve the highest level of accuracy.


# I.	INTRODUCTION 

Compared to the condition of the vehicles and the state of the roads, drivers cause most traffic accidents. The WHO reported an average of 3287 deaths per day, or approximately 1.25 million fatalities annually. Traffic accidents are directly caused by perception errors on the part of drivers. Most collisions are caused by driver error in perception and operation. The behavior of the driver is understood, detected, identified, and predicted with great effort. However, works that are designed to detect driving behavior by integrating accelerometers and gyro-sensors into smartphones fall short if the vehicle is driven in areas with a subpar or, in some cases, nonexistent GPS receiver. Additionally, these systems do not detect aggressive driving, which manifests as rapid, erratic turns, while traveling on a mountain road or a slick road, we brake frequently. Due to the use of complicated and numerous sensors, bio signal-based approaches are not only less practical but also less effective. It is challenging to forecast in single camera-based solutions at night and while we are traveling through a tunnel. There are several detection techniques available to forecast driver behavior, but no comprehensive strategy exists to forecast precise behavior.

We are attempting to assess the driver's conduct using neural networks, and we will be sending updates about his behaviors to his emergency contacts and other passengers through text message. To create a neural network and train the neural network; to train our model, we develop several different techniques. After we have finished training the model, we will evaluate its accuracy. Since it increases the effectiveness of our neural network, analysis accuracy is more crucial. With the help of the train-test split method, we employed 70% samples for training and 30% samples for testing in order to achieve the most efficiency and accuracy possible.

# II.	DEEP LEARNING AND NEURAL NETWORKS

# A.	Neural Network
A neural network is a method of Artificial Intelligence that will help computer learn how to process data just like a human brain.  It uses interconnected nodes or neurons in a layered structure that resembles the human brain. It creates an adaptive system that helps computers to learn from its mistakes and improve continuously. NN will help computers to make intelligent decisions on its own. NN can be used across many industries due it wide variety of use cases. 
There are 3 types of neural networks Feedforward neural network, backpropogation algorithm, CNN. Training a NN is nothing but teaching it how to perform a task. NN process unknown inputs more accurately. They learn by processing large sets of labeled or unlabeled data. It slowly builds it’s knowledge from datasets. Once the neural network is trained properly it will start making educated guesses.

# B.	Deep Learning

Deep learning is a class of machine learning algorithms which will use layers to extract higher level features from the raw input. 


# III.	ALGORITHMS USED

# A.	Stochastic Gradient

In order to comprehend and contrast various approaches to this problem, there have been a number of explorations into machine learning techniques and how they may be applied to the MeSH indexing problem in a more abstract setting 6, 10-15. Then, after testing on the validation set for each training run over the new training set, the number of iterations over the training data that are determined to be optimum is recorded.
One of our contributions to the study (Wilbur et al) is to demonstrate that early stopping can be implemented on a variety of text classification problems by just employing a fixed number of iterations and still get the same performance as the validation set technique. In this study they specifically discover that SGD with a fixed number of eight iterations over the training data delivers MeSH classification results that are on par with early stopping based on held out data and numerous other widely used approaches that need substantially longer to train.
The predictions for a specific document might then be ranked using the SVM scores of all the MeSH keywords for that document. We did not get the optimal result with such a ranking strategy based on raw scores since various MeSH words have varying frequencies in the training data, which results in classifier scores that are not directly comparable. We train classifiers on each half of a given MeSH phrase. We apply the obtained probabilities over all MeSH words to produce ranked MeSH phrase predictions since probabilities are optimally similar.

# B.	Adam optimization Algorithms

In order to update network weights more effectively, Adam optimization, an extension of stochastic gradient descent, can be employed in place of conventional stochastic gradient descent. Adam is ideally suited for non-stationary targets as well as extremely noisy and sparse gradients because to its efficient computation, low memory requirements, and low memory requirements. Very little adjusting is needed.
To converge more quickly, it makes advantage of momentum and adjustable learning rates. By taking the exponential weighted average of the gradients into account, this is also used to speed up the gradient descent procedure.
AdaGrad can be improved using the adaptive learning method RMSprop. Instead of using the sum of squared gradients as in AdaGrad, it uses an exponential moving average. It inherits both the exceedingly complex characteristics of momentum and RMSprop.

# C.	VGG16

A convolutional neural network with 16 layers is called VGG-16. The ImageNet database contains a pretrained version of the network that has been trained on more than a million pictures. The pretrained network can categorize photos into 1000 different item categories, including several animals, a keyboard, a mouse, and a pencil. The network has therefore acquired rich feature representations for a variety of pictures. The network accepts images with a resolution of 224 by 224.
Convolutional neural networks (CNNs) are used initially to categorize pictures. The input images are processed through a number of layers, including convolutional, pooling, flattening, and fully connected layers. After creating CNN models from scratch, we will attempt to fine-tune the model using the approach of picture augmentation. As a result, we will use the VGG-16 model, one of the pretrained models, to categorize images and assess accuracy for both training and validation data. (Tammina et al)


 



# D.	ResNet 50

A convolutional neural network with 50 layers is called ResNet-50. The ImageNet database contains a pretrained version of the network that has been trained on more than a million photos.
The pretrained network can categorize photos into 1000 different item categories, including several poses of the driver, usage of phone, drowsiness based on activities etc. The network has therefore acquired rich feature representations for a variety of pictures. The network accepts images with a resolution of 224 by 224.

Our ResNet 34 model has been somewhat changed with the addition of bottleneck blocks.
A 1x1, 3x3, 1x1-style method has been used in place of our previous 3x3 + 3x3 convolutions, with the final 1x1 convolution having four times as many layers. This expands our network, which enhances performance. The important thing is that this method is likewise inexpensive to analyze, so we receive better outcomes for nearly the same computational cost.
Numerous different techniques in this area can be integrated with the residual approach.
Remaining stacks may be used to mix several convolutional method sets (Cells) to solve various challenges. Many large-scale reinforcement learning methods employ thick stacks of convolutional layers in conjunction with residual networks (AlphaZero is a famous example).

# E.	MobileNet

The input photographs are categorized into pre-established labels or categories by image classification. The classification models gain knowledge from the training picture collection, which eventually aids in prediction. Classifications include binary and multiple labels. Multi-class classification works with more than two labels, while binary classification only deals with two classes/labels.

# F.	Xception:

While RESNET was created with the intention of getting deeper networks, Xception was created for getting wider networks by introducing depthwise separable convolutions. By decomposing a standard convolution layer into depthwise and pointwise convolutions, the number of computations reduces significantly. The performance of the model also improves because of having multiple filters looking at the same level.






# RESULTS & CONCLUSION

Extra Leaves added	Train Loss	Validation Loss	Test Loss
VGG16	No	0.41	0.5	0.64
VGG16	Yes	0.42	9.57	0.58
RESNET 50	No	0.34	0.55	0.57
RESNET 50	Yes	0.32	0.47	0.45
Xception	No	0.4	0.55	0.57
Xception	Yes	0.42	0.52	0.45
Mobilenet	No	0.28	0.4	0.39

	Xception	Resnet50	VGG16	Mobilenet
Safe driving	78.9%	92.5%	54.4%	78.2%
Texting-Right	93.2%	92.3%	96.2%	89.1%
Talking – Right	99.4%	94.3%	95.1%	97.4%
Texting -Left	98.2%	98.2%	98.5%	98.5%
Talking- Left	76.5%	75.8%	87.9%	77.3%
Operating the Radio	94.8%	99.0%	96.8%	97.5%
Drinking	80.4%	86.2%	90.3%	79.3%
Reaching Behind	100%	99.4%	81.1%	99.7%
Hair and Makeup	40.1%	53.3%	63.6%	59.6%
Talking to passenger	58.8%	68.5%	83.1%	79.5

While each of the architectures above gave us good results, there is significant variance in the performance of each model for individual classes. From the table below, we notice that different models have the best accuracy for each class. Hence, we decided to build an ensemble of these models.


# DATA SET GOOGLE DRIVE LINK:

https://drive.google.com/file/d/1Aw7T4Jb3q9tHur4GDCSdHRjpIBopYrcW/view?usp=share_link


# REFERENCES

[1]	H. Lazar and Z. Jarir, "Road traffic accident prediction: a driving behavior approach," 2022 8th International Conference on Optimization and Applications (ICOA), 2022, pp. 1-4, doi: 10.1109/ICOA55659.2022.9934000.
[2]	P. K. K R and S. A M, "Driver Behavior Analysis Using Deep Learning," 2022 International Conference on Computer Communication and Informatics (ICCCI), 2022, pp. 1-3, doi: 10.1109/ICCCI54379.2022.9740778.
[3]	M. R. Othman, Z. Zhang, T. Imamura and T. Miyake, "Modeling driver operation behavior by linear prediction analysis and auto associative neural network," 2009 IEEE International Conference on Systems, Man and Cybernetics, 2009, pp. 649-653, doi: 10.1109/ICSMC.2009.5346668..
[4]	Bhumika, D. Das and S. K. Das, "RsSafe: Personalized Driver Behavior Prediction for Safe Driving," 2022 International Joint Conference on Neural Networks (IJCNN), 2022, pp. 1-8, doi: 10.1109/IJCNN55064.2022.9892982.
[5]	G. Baicang, J. Lisheng, S. Jian and Z. Shunran, "A risky prediction model of driving behaviors: especially for cognitive distracted driving behaviors," 2020 4th CAA International Conference on Vehicular Control and Intelligence (CVCI), 2020, pp. 103-108, doi: 10.1109/CVCI51460.2020.9338665.
[6]	X. -y. Yan, Z. -h. Song, Z. -x. Zhu and E. -r. Mao, "Driver Behavior Failure Probability Prediction Based on CREAM," 2009 International Conference on Information Engineering and Computer Science, 2009, pp. 1-4, doi: 10.1109/ICIECS.2009.5362646.
[7]	Wilbur WJ, Kim W. Stochastic Gradient Descent and the Prediction of MeSH for PubMed Records. AMIA Annu Symp Proc. 2014 Nov 14;2014:1198-207. PMID: 25954431; PMCID: PMC4419959.
[8]	Tammina, Srikanth. (2019). Transfer learning using VGG-16 with Deep Convolutional Neural Network for Classifying Images. International Journal of Scientific and Research Publications (IJSRP). 9. p9420. 10.29322/IJSRP.9.10.2019.p9420.



 


