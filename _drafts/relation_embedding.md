---
layout: default
title:  A relation embedding for ePADD
categories: ePADD
---

# Embedding personal relations in ePADD #

Imagine an email archive with thousands of contacts in the address-book, each one of them have a special relation with the owner of the archive which is reflected by the content of their conversation.
For example: colleague, family, student, teacher, mentor, employer etc.
We would like to assign an embedding to each os such relation so that we can do wizardly things like finding contacts that show the same relation as a given pair, describing a relation with words etc. through vector space operations.

An interesting lead is the work by Mohit Iyyer's [finding relations between characters in creative content][rel-novels]. 
They propose a method that finds the dynamic/evolving relation between characters in a creative text such as novel.
For any two given characters, text appearing in between the mentions is collected in chronological order. 
All such words are simply assigned the embedding equal to the average of them all. 
The problem is then formulated as to assigning a relation vector to each of such embedding across time. 

Special care is taken to ensure the following:
  * A regularization term $$\Vert RR^T-I\Vert $$ is added to keep the relation vectors from being similar to each other.
  * The relation embedding are forced to be in the same vector space as the word vectors, hence it is possible to interpret them.
  * Word vectors are used, importing the available external knowledge
  
Because of the above reasons: the method is bound to work better than topic models.
  
This image, from their publication, summarizes the approach well. 

![RMN Architecture](/assets/images/mohit-rmn.png)

It's a single layered neural net with randomly initialized R, C, B matrices which correspond to the relation matrix of [Number of relations]x[Relation Embedding size], character matrix of shape [Number of characters]x[Character Embedding size] and vector of size [Book Embedding]. 
Each of the embedding are fixed to have the same size as the word embedding.
*f* is a ReLU function.
The extra complexity in the expression for $$d_t = \alpha\cdot softmax(W_d\cdot[h_t;d_{t-1}]) + (1-\alpha)\cdot d_{t-1}$$ is to allow for smooth transition between the relation weights (states: $$d_t$$).
Finally, $$r_t$$ and $$v_{s_t}$$ are constrained to be similar with a hinge loss.

$$J(\theta) = \sum_{t=0}^{|S_{c_1,c_2}|}\sum_{n \in N} max(0, 1-r_t v_{n}+r_t v_{s_t})$$

where N is a set of negative samples randomly sampled from other spans.


## Through the lens of ePADD ##

There are some aspects which are not relevant in the case of emails
  
  1. The relation between characters is not as dramatic as it would be in a novel, which makes the concept of relation trajectory redundant, I think.
  2. The character matrix and book matrix provide a cushion for when the context is too sparse to imply the relation. If the relation trajectory is given up, we are finding a relation that explains best every bit of context that is available, hence these matrices can be given up too.
  3. There is a higher quality cue for when the characters interact; Any email with both the characters in to/from. The entire document is the span and averaging over them all is a little careless, as we are throwing away some important information such as the position of the word in the document and the words appearing in a sequence (The way the person is addressed: "Hey Bob" to "Dear Prof. Creeley", how the email is signed off, if it ends in a "Thanks" can say a lot about the relation)
  4. The relations are not always symmetric; Professor-student, Father-daughter etc. For this reason, the c1->c2 and c2->c1 relations should be distinguished.

We revise the model to include the above observations.

The embedding of a document (email) is given by an LSTM cell over embeddings of all the words in the document.
Since there is no book and character matrices; $$h_t = f(W_h\cdot[v_{s_t}])$$.  
Because there is no relation trajectory: $$d = LSTM(h_1\dots h_T)$$ and $$r=R^T d$$.
$$J(\theta)$$ will remain the same except that the subscript t on r will be dropped.

As a first attempt at it, I tried a simple method of embedding a document: averaging the embedding vector of all the words mentioned in the document (This is the one proposed in the paper linked above).
This is no good, I have tried it on my own mail (sent mail) and looked at the closest documents based on the embeddings so assigned; It made no sense, whatsoever. 
Given this, it is unsurprising that the relations learned and the relation weights assigned to documents also made no sense. 
I think it is vital that an aggregator is trained end-to-end which can identify key-words better.

[rel-novels]: http://cs.colorado.edu/~jbg/docs/2016_naacl_relationships.pdf "Feuding Families and Former Friends: Unsupervised Learning for Dynamic Fictional Relationships"
