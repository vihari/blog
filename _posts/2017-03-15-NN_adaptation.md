---
layout: default
title:  Literature Survey on Adaptive Learning in the context of neural networks
categories: deep-learning speaker-adaption adaptive-neural-networks
---

# Introduction

Adaptation of neural network models in the context of speech, speaker adaptation, is well researched. 
Acoustic models have evolved starting from GMM or HMM models to hybrid models that pushed in to the neural network regime, making way for deep neural networks to LSTM based RNN models.
The problem of low performance due to train and test domain differences is acute in speech and is well addressed in the past even before the introduction of neural networks.
In this post, I will discuss the domain adaptation problem only in the context of neural nets.

## [Speaker Adaptation of hybrid NN/HMM based on speaker codes][speaker-code]

In a NN/HMM model, the GMM component that is used to estimate the emission probabilities of each state is replaced by neural network which when fed with the feature vector, outputs the posterior over HMM state labels.
	
   ![speaker-code-fig1](/assets/images/speaker-code-fig1.png)

As shown in the image above, the proposed adaptation method relies on learning adaptation NN and speaker codes.
All the layers in adaptation NN are standard fully connected layers.
The transformed feature vector should have the same dimension as the input feature vector.
During the adaptation phase, both the weights of adaptation NN and speaker need to be learned.

### Training

The training of speaker independent model and adaptive NN along with speaker code is carried in two separate steps.
Speaker Independent (SI) model is learned as if adapt NN does not exist, that is standard NN-HMM model is trained without using any speaker specific information.  
**Once the SI model is trained, its weights are freezed and speaker code and adaptNN weights are learned jointly with back-propagation to optimize frame-wise classification performance.**  
Acknowledges that there are several other plausible ways of training the network, but do not provide any rationale as to why this method was choosen as such.
  * it is possible to tweak the weights of SI model when optimizing for speaker codes and adaptNN weights.
  * Learn all the three parameters jointly. "However, this may result in two inseparable NNs and they eventually become one large deep NN with only a number of lower layers receiving a speaker code."
  * Another possibility is to learn SI model over the features transformed with an already trained adaptNN, that is to flip the order in which we train.

### Adaptation

During this phase, only the speaker code is to be learned for the new speaker over the small amount of data available for adaptation.
This is one of the strong points of this work that the speaker code can be arbitrarily made small or large depending on how much data is available for adaptation.
We do the same, BP, but adaptNN weights are freezed as well.

### Interesting experiments

![fig2](/assets/images/speaker-code-fig2.png)

The figure above shows the effect of number of examples used for estimating speaker code on performance.  
The experiment was conducted on a 462 speaker training set and 24-speaker test set.
The test set each contain eight utterances per user.
The learning rate, context window are al fixed, hidden layer width (1000), speaker code size (50), 183 target class labels and feature vector dimension (??) are all fixed.

Note
  * "Dummy" is when no speaker code, but only dummy layers -- does not affect the performance meaning that the perf. improvement is not just increased model complexity.
  * using zero adaptation has some positve effect.
  * Even when exposed to one utterance, the perf. improvement is not bad.

## [Using I-Vector inputs to improve speaker independence][]

Leveraging utterance-level features as inputs to DNN to facilitate speaker, channel and background normalization.

### i-Vectors or identity vectors

**"i-vectors encode precisely those effects to which we want our ASR system to be invariant: speaker, channel and background noise."**
These vectors are generally used in speaker recognition and verification

### Adapting with i-vectors

![google-ivec-fig1](/assets/images/google-ivector-fig1.png)

As shown in the image above, the idea is to provide the input with characterisation of the speaker, which could enable it to normalise the signal with respect to speaker specific nuances and thus leading to a better Speaker Independent model.

### Experiments

The training and dev set performance differed when the input is compounded with the 300 dimensional i-vector. 
This could mean that the network is over-fitting the i-vectors or it could also be that the computing a 300-dim vector from short utterances is not relaiable.  

Reducing the i-vector dimension, to say 20, along with l2 regularization helped. 

The dataset contains 80 speakers with an equivalent of 10 minutes of utterance per user.
The input augmentation with 20-dimensional i-vector model along with re-training on the adaptation set with l2 reg. coeff of 0.01 improved the results further. 

![ivec-adapt-results](/assets/images/ivec-adapt-results.png)

This work claims that when the network with input augmented with i-vectors is also adapted over the user-data, it can lead to better performance as shown in the figure above.
It is not mentioned if any weights are fixed while adapting, it is probably that none of the weights are fixed and are all jointly optimized over various passes on the user data.
The baseline model is a normal feed-forward neural network.

## [Speaker adaptive deep neural networks][] ##

Two different model architectures are tried with: AdaptNN and iVecNN.

![models](/assets/images/ivecNN.png)

[//]: # AdaptNN arch. is very similar to the fast adapt work except that i-vectors are used instead of speaker codes which mean that no adaptation is required.
[//]: # Also, the i-vector is not fed to the final transformed features, which the authors claims to be better performing.

iVecNN works by producing a linear feature shift which is added to the original feature vector and is activated with a linear activation function.  
$$ a_t = o_t+f(i_s) $$  
The weights of iVecNN are estimated using BP while the weights of the initial DNN are estimated and fixed.

The strong point of this method is its relevance to CNNs.

In the figure above, $$z_t$$ is the element-wise sum of $$o_t$$ and $$y_s^{-1}$$.
For two speakers in the training set, two principal components from PCA are plotted as shown in the image below. 
Observe that the non-overlapping regions has shrunk for the case of $$z_t$$ when compared to $$o_t$$ implying that adding a linear shift to the original vector is actually making the speaker independent.

![speaker independence](/assets/images/speaker-indep-ivecNN.png)

The training pipeline of the system is shown below. 

![pipeline](/assets/images/ivecNN-pipeline.png)

They did show that the model is better than DNN+i-vector, that is the augmented input, the first model in this article, and concluded that this model is better than the other.
However, the experiment was not set right, the i-vectors are not normalized and i-vector size is not varied.
I am not including results because I did not like them. (I have some issues with how the experiment was set-up)

# Adaptation in the context of hand writing recognition

I have not come across any work that adapts hand writing recognition to a user. 
There is some interest in recognizing what is called as unconstrained hand writing recognition task which is recognizing hand written characters with no restrictions imposed on their style, size, position and medium.

## [Generating Sequences With Recurrent Neural Networks][gen-hw]

This work is an interesting read, although it does not explicitly make user modeling.
The model can generate hand writen sentences that resemble the ones written by human.
Also interesting is that it is possible to tune the style of the generated sentence instead of randomly choosing one.
This work demonstrates that it is possible to generate such writings one point at a time with RNNs that are also consistent with a style. 

Unlike others, no pre-processing of the data (online data) is made.
According to them, pre-processing will normalize and removes variance in the input and will lead an output that is more synthetic.

## [A Novel Connectionist System for Unconstrained Handwriting Recognition][garves-09] ##

## [Meta-Learning with Memory-Augmented Neural Networks][memaug-one-shot] ##
Given a small amopunt of data, a straightforward gradient based solution is to completely relearn the parameters from the data available, which can lead to poor learning due to interference.
One-shot learning is quite hard because of the interference effects due to the training params learned over much larger data.
`Architectures with augmented memory capacities, such as Neural Turing Machines (NTMs), offer the ability to quickly encode and retrieve new information, and hence can potentially obviate the downsides of conventional models.`



[memaug-one-shot]: http://jmlr.org/proceedings/papers/v48/santoro16.pdf "Meta-Learning with Memory-Augmented Neural Networks"

[Using I-Vector inputs to improve speaker independence]: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6853591 "IMPROVING DNN SPEAKER INDEPENDENCE WITH I-VECTOR INPUTS"

[speaker-code]: http://ieeexplore.ieee.org/document/6639211/ "Fast speaker adaptation of hybrid NN/HMM model for speech recognition based on discriminative learning of speaker code"

[Speaker adaptive deep neural networks]: https://www.cs.cmu.edu/~ymiao/pub/tasl_sat.pdf
 "Towards Speaker Adaptive Training of Deep Neural Network Acoustic Models"

[gen-hw]: https://arxiv.org/pdf/1308.0850v5.pdf "Generating Sequences With Recurrent Neural Networks"

[garves-09]: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4531750 "A Novel Connectionist System for Unconstrained Handwriting Recognition"


