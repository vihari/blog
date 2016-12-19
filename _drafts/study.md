---
layout: default
title:  "Notes from my independent study of deep learning"
date:   2016-10-25 16:42:25 +0530
categories: jekyll update
---
## Practical aspects of Training ##
	

  * [Recommendations by Y. Bengio][rec-bengio-2012]  
  The recommendations are not definitive and should be challenged.
  They can work as a good starting point.  
  It has been shown that use of computer clusters for hyper-parameter selection can have an important effect on results. 
  The value of sopme hyper-parameters can be selected based on its performance on training data, but most cannot. For any hyper-parameter that affects the capacity of the learner, it makes more sense to use an out-of-sample data to select the params, hence the validation set.  
  	> different researchers and research groups do not always agree on the practice of training neural networks
	
	* **Initial Learning rate ($\epsilon_0$)** is the single most important hyp. param. according to him. `If there is only time to optimize one hyper-parameter and one uses stochastic gradient descent, then this is the hyper-parameter that is worth tuning.` Typical values for neural networks is: 1E-6 to 1 that is if the input is properly normalized to lie between 0 and 1. A default value of 0.01 typically works for multi-layer neural network.
	* **Learning rate schedule**: The general strategy is to keep the learning rate constant until some iterations as shown in the equation below. In many cases choosing other than the default value for $$\tau\rightarrow \infty$$ has very little effect. 
	  There are suggestions to decrease the learning rate less steeply than linear after $$\tau$$ time steps that is $$O(1/t^\alpha), \alpha<1$$ depending on the convex behaviour of the function being optimized.
	  Adaptive strategies for learning rate exist.
	  
		$$\epsilon_t=\frac{\epsilon_0 \tau}{max(t,\tau)}$$
	* **Mini-batch size (B)**: In theory, this hyp. param. should only affect the training time and not performance. Typically, B is a value between 1 and a few hundred. A value of 32 is good because it can take implementational advantages that matrix-matrix product are more efficient than matrix-vector and is not too high to require a long training time. 
	  B can be optimized independently of others. Once a value is choosen, it can be fixed and is better to re-optimize in the end since it weakly interacts with other hyp. params.
     * **Number of Training iterations (T)**: To be choosen by early-stopping criteria. During the analysis stage where we try and compare different models, stopping early can have evening out effect. That is we cannot make-out between an over-fitting or under-fitting model. For this reason, it is best to turn it off during analysis. Another param called *patience* is generally defined that tells how long the model should wait after it observed the minimum validation error, the param is defined in terms of minimum number of examples to be seen before stopping.
     * **Momenum ($$\beta$$)**: $$\hat{g}\leftarrow (1-\beta)\hat{g}+\beta g$$ where $$\hat{g}$$ is the smoothed gradient. The default value of $$\beta=1$$ works for most cases, but is found to help in some cases of unsupervised learning. "The idea is that it removes some of the noise and oscillations that gradient descent has, in particular in the directions of high curvature of the loss function"
	 In some rare cases, layer-specific hyper paramater optimization is employed. This makes sense when the number of hidden units vary large between layers. Generally employed in layer-wise unsupervised pre-training.  
	 
	 The parameters discussed above are related to SGD (Stochastic Gradient Descent), what follows are the params related to the neural network.
     
	 * **Number of hidden units $$n_h$$** Because of early stopping and possiblty other regularizers such as weight decay, it is mostly important to choose $$n_h$$ large enough. In general, it is observed that seeting all the layers to equal number of units works better or the same as when setting the hidden unit widths in decreasing or increasing order (pyramidal or reverse pyramidal), but this could be data-dependent ([Larochelle et.al. 2014][larochelle2014]). 
	 An over-complete (larger than the input vector) is better than an under-complete one.
	 A more validated observation is that optimal $$n_h$$ value when training with unsupervised pre-training in a supervised neural network. Typically from 100 to 1000. This could be because unsupervised pre-training could hold lot more information that is not relevant to the task and hence require large hidden layers to make sure relevant information is captured.
	 * **Weight decay:** This article makes a very inetersting note about thsi parameter. As we know, weight decay is used to avoid over-fitting by limiting capacity of the learner. L2 or L1 regularization correspond to the penalties: $$\lambda \sum_i {\theta_i}^2$$ and $$\lambda \sum_i{\theta_i}$$, both the terms can be included. In the case of batch-wise handling of data, the param $$\lambda$$ that we optimize is actually $$\frac{\lambda'*B}{T}$$ where B is batch size and T is the size of training data.
	   L2 regularization corresponds to a guassian prior (over weights) $$\propto exp^{\frac{-1 \theta^2}{2 \sigma^2}}$$ with $$\sigma^2=1/2\lambda$$.
	   *Note that there is a connection between L2 regularization and early stopping with one basically playing the same role as other*. L1 regularization is different and sometimes act as feature selectors by making sure the parameters that are not really very useful go to 0. L1 corresponds to laplace density prior $$\propto e^{-\frac{|\theta|}{s}}$$ with $$s=\frac{1}{\lambda}$$  
		It is sufficient to regularize just the output weights in order to constrain the capacity. (we use the input weights and output weights to denote weights corresponding to the first and last layer. The input weights is also often referred to as *filters* because of analogies with signal processing techniques)  
		Using an L1 regularizer helps to make the input filters cleaner and easier to interpret. 
		We may draw that L1 cleans the input weights and L2 the output weights. 
		When we introduce both the penalties into our optimization, then it is required to tune the coeffs for L1 and L2 independently. In particluar, input and output weights are treated different.  
		*In the limit case of the number of hidden layers going to infinity, L2 regularization corresponds to SVMs and L1 to Boosting ([Bengio et.al. 2006][Bengio2006a])*  
		One of the reason why we cannot rely only on early stopping criteria and treat input and output weights differently from hidden units is because they may be sparse. For example, some input features could be 0 more frequently and others non-zero more frequently. A similar situation may arise when target variable is sparse i.e. trying to predict a rare event. In both cases, the effective number of meaningful update (active feature or rare event) seen by these params is less than the actual number of updates. The parameters (weights outgoing from the corresponding input) of such sparse examples should be more regularized, that is to scale the reg. coeff. of these params by one over effective number of updated seen by the parameter. This presents an alteranaate way to deal with imbalanced data or anamolies(?).
	* There are several approaches that aim to minimize the sparsity of hidden layers (note that sparsity is very different from L1 norm)
    * **Non-linear activation functions**: Popular choices are: sigmoid $$\frac{1}{1+e^{-a}}$$, the hyperbolic tangent $$\frac{e^a-e^{-a}}{e^a+e^{-a}}$$, rectifier max; max(0,a) and hard tanh.  
	  Sigmoid was shown to yield serious optimization difficulties when used as the top hidden layer of a deep supervised network without unsupervised pre-training. 
	  `For output (or reconstruction) units, hard neuron non-linearities like the rectifier do not make sense because when the unit is saturated (e.g. a < 0 for the rectifier) and associated with a loss, no gradient is propagated inside the network, i.e., there is no chance to correct the error. For output (or reconstruction) units, hard neuron non-linearities like the rectifier do not make sense because when the unit is saturated (e.g. a < 0 for the rectifier) and associated with a loss, no gradient is propagated inside the network, i.e., there is no chance to correct the error` 
	  The general trick is to use linear output and squared error for Gaussian output model, cross-entropy and sigmoid output for binomial output model and log output[target class] with softmax outputs to correspond to multinomial output variables (that is take softmax for over all the neuron outpus and score by considering only target label required).
    * **Weights initiazation and scaling coefficient**: The weights should be initialized randomly inorder to break symmetry and bias can be initialized to 0. 
	  If not all the neurons initially will produce the same output and hence receive the same gradient, wasting the capacity.
	  The scaling factor controls how small or big the initial weights are. Units with large input (fan-in of the unit) should have smaller weights (I have first-hand experience with problems that arise when this is not done with one of my NN assignments. The initialization that worked smaller number of hidden units just over-shooted due to explosive gradients. That is because the output diverged to a large value when I did this)  
	  The recommendation made is either to sample *Uniform(-r,r)*. 
	  Where r is $$\sqrt{\frac{6}{fan-in+fan-out}}$$ for hyperbolic tangent units and  $$4*\sqrt{\frac{6}{fan-in+fan-out}}$$. fan-in and fan-out are the input and output dimension of a hidden layer.
  
  General advice on finding the best model.  
  Numerical hyper-parameters need to be grid-searched in order to find one. 
  It is not sufficient to conclude the best value based on comparison with less than 5 other values.   
  Scale of values considered is often an important decision to make, this is the starting interval in which the values will be looked up. 
  It makes more sense to sample values uniformly in the log space of such interval than to blindly evaluate at every value because the perf. at say 0.01 and 0.011 is likley to remain the same.  
  Strategies for hyper-param selection: Coordinate descent and Multi-resolution search.
  Coordinate descent: Make changes to the each hyper-param one at a time, find the best value for the param and move on to the next one. 
  Mult-resolution search: There is no point in fine-tuning or high-resolution search over large intervals. Do a low-resolution search over several settings and then high-res search over best configurations.
	  
## Limitations of Deep Learning ##
  * [DNNs are easy to fool][easy-fool-dnn] Surprisingly, DNNs can be easily fooled by adding adverse perturbations to an image. For example, adding noise that is imperceptable to humans to an image that looks like *panda* and recognized as one with confidence of ~56% will lead to an image that is wrongly labeled but with very high confidence. This paper details about very interesting case-study of where the state-of-art AlexNet utterly fails. For code and images [see][easy-fool-dnn-site]. There has also been some effort at explaining this phenomena. In the paper: [Explaining and Harnessing Adversial Examples][Ian-why-easy-fool] by Ian Goodfellow et. al., they make a case that such a thing happens majorly because the DNNs are linear in nature.

## Wisdom from random sources ##

### [Hinton at Stanford][geoffrey-stanford] ###

* Big data is good (something that frequentist statisticians suggest)   
  * For any given size of model, its better to have more data
  * But it's a bad idea to try to make the data look big by making the model small 
* Big models are good (Something that statisticians do not believe but true)
    * For any given size of data, the bigger the model, the better it generalized, provided you regularize it well.
	* This is obviously true id your model is an ensemble of smaller models. Adding extra models to the ensemble alwqays helps.
	* It's a **good idea** to try to make the data look small by using a big model.
 * Dropout enables the all the models in an ensemble to share knowledge. If there is only one layer in the network with with H, then the model with dropout is an ensemble of 2^H models and softmax over the output layer is geometric mean of all the models in ensemble.   
 * Dropout can be seen as bernoulli noise, we do not change the expected value because a neuron either emits zero or twice the value. It is noted that any other kind of noise can work just as well. Gaussian noise and Possion noise are tested to give same performance if not better. In these cases a multiplicative noise with standard deviation equal to the activity. The point is that neurons do not share real values but spikes and that is a lot better than trhe actuaol values.
 * In this lecture, Hinton goes on lengths elaborating why brains cannot do exact back-propagation and explains possible other ways in which it could be learning the weights. He argues that neurons in a feed-back loop can do away with the need to back-propagate by considering the difference between the inout at this instance and previous one (plasticity of brain, Spike-time dependent plasticity)

[easy-fool-dnn]: https://arxiv.org/pdf/1412.1897.pdf
[easy-fool-dnn-site]: http://www.evolvingai.org/fooling "Deep neural networks are easily fooled: High confidence predictions for unrecognizable images "
[Ian-why-easy-fool]: https://arxiv.org/pdf/1412.6572v3.pdf "Explaining and Harnessing Adversial Examples"
[rec-bengio-2012]: https://arxiv.org/pdf/1206.5533v2.pdf "Practical Recommendations for Gradient-Based Training of Deep Architectures"
[larochelle2014]: http://deeplearning.cs.cmu.edu/pdfs/1111/jmlr10_larochelle.pdf "Exploring Strategies for Training Deep Neural Networks"
[Bengio2006a]: https://papers.nips.cc/paper/2800-convex-neural-networks.pdf "Convex Neural Networks"

[geoffrey-stanford]: https://www.youtube.com/watch?v=VIRCybGgHts "Can the Brain do back-propagation?"

