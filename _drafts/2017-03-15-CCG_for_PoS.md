---
layout: default
title:  Notes from "Dan Garrette: Exploiting Universal Grammatical Properties to Induce CCG Grammars"
categories: notes CCG PoS
---

The link to the talk can be found at: https://www.youtube.com/watch?v=RfBBcI7DbL4

In this talk, Dr. Garrette presents a weakly supervised approach to PoS tagging, especially for languages with sparse resource. 
The input to the system is unlabelled data along with tag dictionary which lists all possible tags for words.
It is claimed that the accuracy of tagging with the sparse resource (the tag dictionary that is built in a span of 4 hours) is around 90\% for english when compared to 69\% with EM.

CCGs are combinatorial categorical grammars.
They assign categories to each token which are then combined that leads to the parse of the sentence. 
The presenter argues that CCGs are better at generalizing across languages and require less domain knowledge.
It is know that when every  token in a sentence is assigned a tag, the parse tree follows without confusion, hence making the problem into just a sequence tagging problem.
These are called supertags and are like: np/n, S/n etc., when this compqared to the PoS tags, they offer natural advantage that we know that *np/n* goes with *n* and not *np*.
On the other hand, background knowledge is required to know if *DET* combines with *NN*.
CCG supertags offer: universal, intrinsic grammar properties.

The best acuuracy with this appraoch is only 90 because this lacks the frequency information that fully-supervised data contains.
