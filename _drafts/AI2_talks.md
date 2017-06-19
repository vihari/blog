---
layout: default
title: Summaries of interestesting talks
categories: AI2
---

# In this post, I try to summarize some technical talks I found interesting #

### [Josh Tenenbaum: Engineering & reverse-engineering human common sense][ai2-tanenbaum] ###

In this talk, [Josh](http://web.mit.edu/cocosci/josh.html) presented work from his lab of models on how we are able to learn so much so fast.
He talks about "common sense core" which is a collection of theorems which help us perceive lot more than a typical bot. 
For example, knowing just the newton's laws of motion can help us understand a lot of things. 

He points out that although the present day computers learn from large amounts of data (deep-learning), infants learn like the way Galileo, Kepler etc. experimented in medieval times. 

*The experiments on developmental psychology in infants he described in the talk are very interesting. How do you establish that an infant who is 8-12 months old know something when they cannot communicate? They show them images that looks surprising and that which look normal, they tend to stare a lot more on the image with surprising outcome. Also, if they are shown an image in which a toy car flies off a ramp, they play with the car to see if it can really fly by throwing it in the air etc.*

He made a point to the $$AI^2$$ audience that what the big companies trying to do with big data is to serve short-term goals, but at places like this they can afford to have long-term vision and start from scratch by engineering common sense.

### How do we engineer common sense? ###

With Probabilistic Programs (probmods.org).
The idea is simple: just like Bayesian net which is an undirected graph over random variables, each random variable here is a program and a fully defined program defines a forward pass over the network.

[ai2-tanenbaum]: https://www.youtube.com/watch?v=hfoeRiZU5YQ "Josh Tenenbaum: Engineering & reverse-engineering human common sense"

