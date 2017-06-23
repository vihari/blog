---
layout: default
title: Geocoding of documents
categories: AI2
---

# Geocoding of text documents

The relevance of the search results can be improved, especially for the queries with spatial intent such as "restaurants" and "places to visit", by filtering the results based on their focus location and user's location.

Given the application in search engine, unsurprisingly, there are several approaches to annotate the focus of the webpage or geocode every mention in the document.
This document is a summary of the exiting techniques.

## [Web-a-Where: Geotagging Web Content][webawhere]
Is a classic on Geotagging a document's content published in '04 from IBM Labs

The components involved in the pipeline are (1) spotting (2) disambiguation and (3) determining page focus
The entities are spotted by matching through gazette lookup.
The gazette is a collection of various sources and contains, state codes, acronyms, countries, city in a hierarchy.
To avoid spurious matching of location entries such as "Humble", "AT", "IN", "Of", "To"; several heuristics based on capitalization and popularity of the location are used to avoid matching them.

Disambiguation is just a bunch of heuristics:- is of two kinds: between geo and non-geo terms, for example: London (Person), London (Place) and between geo entities for example London_(England), London_(Canada).

The rules are as below
  * If a token can be resolved properly in conjunction with neighbouring ones, it will be
  * The meaning of a qualified entity is propagated to all the referent mentions in the document.
  * For any other unqualified entity, a context from the document is induced for resolution. If the doc contains "London" and "Hamilton" which are contained in "England, UK", "Ontario, Canada" and "Ohio, USA" and "Ontario, Canada" respectively. "Canada" would be the answer based on frequency.
  
This is then followed by finding the focus/foci of the page.
This again is a simple algorithm.
Once all the spots are disambiguated with confidence scores assigned, the top scoring resolution for each of the mentions is assigned a score=square of confidence and a subsequently decaying score for each of the enclosing regions.
The scores are accumulated, and with threshold and some care in selecting the focus, they are reported.

They have evaluated the technique on corpora from three different domains: arbitrary (web)--popular pages, pages crawled from ".gov" domain, Open Directory Project (ODP) pages.
The mentions are hand-labelled for ground truth, this is for evaluation of disambiguation (do not seem to have made publicly available the data). 
Evaluation of focus is much simpler.

## [Computing Geographical Scopes of Web Resources -- VLDB 2000][vldb00]
This paper is also about identifying the focus of web resources. 
Along with the content of the web page, they also look at the spatial spread of the hyperlinks to/from the page in order to determine the Geo scope.
Here again, the disambiguation technique is quite simple.
The process is guided by unambiguous mentions in the document.
If the document contains 90% of unambiguous mentions in Newyork and 10% in Washington, then every other ambiguous mentions is assigned confidence based on this.

They have avoided the disambiguation problem by assigning confidence this way.
Which is fine for them because they are only interested in the geographical scope.

## [Detecting geographic locations from web resources][gir04]
By a team at Microsoft (Research) published in '04 in Geographic Information retrieval.
This paper is terribly written and looks bogus; I do not want to spend too much time with this.
The approach goes about finding the dominant location in the web resource which explains most location mentions in it by either the location itself or through its off-spring. 

## [Large-scale Location Prediction for Web Pages][tkde17-clicklogs]
This work, published in '17 at TKDE by a team at Yahoo!, finds the focus of a page by building term vectors for lexical tokens using click logs.
Term vector reflects the probability distribution of the token with different geographical locations.
Term vectors are built in a three step process: (1) query url mapping is mined from the logs. (2) location url map is extracted from queries which contain mentions of location (3) location term map fro each of the token present in the url from the second step.
In the point 2, a logistic regressor is trained to classify a query with and without geograophic scope. 
Geo-queries can be either the ones with explicit mention of location or are implicit, for example "restaurants".
The features for the regressor are either unigrams or ngrams -- I am not sure how the labeled data is obtained.
Once a query is classified as geo, either explicit mention is recognized or user's location is marked as the one.
They further explain that they do not much care how the classifier or location extraction perform since they assimilate over several urls, the noise can go away.

They have further presented the effect of feature (unigram vs ngram etc.) and compared it with baselines, stanford-ner, yahoo-pm etc. the details are way too boring for me to read through.

## [Application of Click Logs in Geosensitive Query Processing][shrotri-report]
This is a Masters thesis that exploited click logs to disambiguate locations.

In a nutshell, the formulation is below:

$$
	e = \operatorname*{argmin}_i \sum_{j\in D}{(sim(C^d, C^{d_j})-sim(E^d+e_i, E^{d_j}))^2}
$$

$$
	sim(S_1, S_2) = \frac{\sum_{i\in S_1}{\sum_{j\in S_2}{[[dist(s_i, s_j)<threshold]]}}}{|S_1|*|S_2|}
$$

Click logs are collected from Wikipedia pages for location topics and for each click also available the IP address of the where the click originated from which serves as the user's location.
E is the set of all location entities mentioned in the document.
$$e_i$$ is the disambiguation under consideration for a mention 'l'.
D is the set of all documents that mentions 'l'.
dist is the euclidean distance measure.

The intuition is to find the disambiguation that makes the click logs and entities look the similar.
Note that this enables even for pages with several geo-scopes. 
For example, if the page is about Sino-Indian war, we expect half the mentions to be in China and some in India, the click logs also, supposedly, would reflect such scattering.

## Conclusion
Most of the papers above took a shot at location mapping only in the interest of finding the focus of the page, and hence are not serious about the disambiguation problem. 
These are some limitations of the existing work.
  * The granularity of the recognized locations is either limited to the Wikipedia pages ([Masters thesis][shrotri-report]) or cities, states and countries ([Web a Where][webawhere] in particular).
  * Since the granularity is limited, all possible entity completions can be obtained by a simple lookup in the gazette which is not scalable. Imagine obtaining all possible unambiguous completions for "Dominos Pizza" or "State Bank of India".

Taking a little detour from Geo-coding, a lot of research went into linking entities.
Of particular interest to me are the ones that does query expansion of entities in order to disambiguate them.

## [Linking Entities to a Knowledge Base with Query Expansion][emnlp11]
This work published in EMNLP '11 exploits the idea of entity expansion to generate several different variants of a mention to help find all candidate linkages.
For example, NYC is expanded to New York City using Wikipedia redirects, *Harry* is expanded to a more complete version: "Harry Potter" mentioned in the same document. 
They also use the query expansion idea to augment the entity with the local/global context words to be able to disambiguate.

### Query expansion for generating candidate entities. 

If the mention is a under-specified one like *Mark (Person)*, or *Rampur (Place)* then there might be too many matches for these spots in the knowledge-base.
For this reason, they are augmented with the local context.
In the case of *Person* and *ORG*, variants are generated from other named entities in the document whose sub-string is the spot.
In the case of *place*, the document may contain the mention of state, country, region subsuming the spot's location. 
For this reason, the variants for the spot is the set of concatenation with each of the named location entity in the document.

Below is an example of generating variants for place related spot.

**Query name string**: Mobile  
**Query document**: The site is near Mount Vernon in the Calvert community on the Tombigbee River, some 25 miles (40 kilometers) north of Mobile. It's on a river route to the Gulf of Mexico and near Mobile's rails and interstates. Along with tax breaks and 400 million (euro297 million) in financial incentives, Alabama offered a site with a route to a Brazil plant that will provide slabs for processing in Mobile.  
**Alternative Query Strings**:
Mobile, Mobile Mount Vernon, Mobile Calvert, Mobile River, Mobile Mexico, Mobile Alabama, Mobile Brazil  
[Example copied from the paper]

Also, they have used an external knowledge-base such as redirects table from Wikipedia to obtain possible variants for names like *Master Blaster* for *Sachin Tendulkar* and also to help with acronyms.

### Query expansion for ranking candidate entities

Given a query Q and the entity in the knowledge-base E, they use KL divergence to measure the distance and score. 

$$s(E, Q) = -Div(E, Q) = -\sum_{w\in V}{P(w/\theta_Q)log(\frac{P(w/\theta_Q)}{P(w/\theta_E)})}$$

Where $$\theta_Q$$ and $$\theta_E$$ are query and entity language models, basically they are multinomial probability distributions.
The idea is to enhance the query and entity language model with the local/global context to do better at ranking.

They borrowed the idea of query expansion with relevance feedback from Information retrieval in order to expand the query language model.

$$P(w/{\theta_Q}^{L}) = \alpha p(w/\theta_Q) + (1-\alpha)p(w/\theta_{D_Q})$$

They have experimented with some variants on how the document's language model is generated: just the named entities, named entities weighted based on position from the spot.

Using the query expansion idea, they have reported a better recall when just using the blind match with just the entity mention.

Also, the augmentation of the query language model with the local context did better at disambiguation.

## [Automated Geocoding of textual documents: A survey of current approaches][geocode-survey]
A survey published in GIS proceedings of 2017.
This is a survey on methods to assign a location to a textual document and it mostly pertained to assigning location to short text like tweets.
The task of assigning a location to a given document is handled very differently from linking each entity mention.
Since linking each entity mention is not of interest, in a nutshell, none of the methods exploit the spatial proximity of linked entities to disambiguate.

Apart from Web-a-Where and, Woodruff and Plaunt, another classic from '94, the survey focused on mostly language modelling approaches.
The language modelling techniques aim to learn a language model corresponding to each geographic region, border-lining computation social linguistics.
While some models learn markovian models over the language, some other went on to finding the geo-related words to the extent that Adams and Janowicz (2012), trained a Latent Dirichlet model (LDA) to find latent topics over corpus of documents D associated with geo-spatial coordinates.
The closest they got to exploiting the spatial proximity is at sampling of words based on geographical density, more formally:

$$GeoDen(w_i) = \frac{\sum_{C_j\in C^{w_i}}{P(C_j/w_i)}}{|C^{w_i}|*\frac{\sum_{\{C_j, C_k: C_j\in C^{w_i}, C_k\in C^{w_i}, j\neq k\}}{dist(C_j, C_k)}}{|C^{w_i}|-1}}$$

The scoring function, GeoDen(w), is used to identify words that are geo-indicative. 

They next discussed discriminative classifiers to do the same and found the to be better.
In the context of the problem for which I am doing literature survey for, this abridged version is of my only interest.

## [Location Name Disambiguation Exploiting Spatial Proximity and Temporal Consistency][socialnlp15]
This is a paper published in 2015 at *SocialNLP 2015@NAACL-HLT*.
I found this paper a little loose and hence doubt its contribution and completeness. 

In the related work section, they do not present any work that also exploits spatial proximity, according to them it is a novel feature.
This section focused half the time on models that find location sensitive words based on *Information Gain* or *GeoDen*(above) like metrics and another half on works that made good use of locations from tweets as a motivation.

They generated a training dataset with some high accuracy heuristics on Tweets with GIS data.
They have collected all possible locations and location mentions from Japan Wikipedia. 
For each location variant, they have an SVM classifier trained which finds the right entity.
The features for such a classifier are lexical: bag of words and frequency of other location entities in the document (tweet) along with spatial proximity features. 
They are the distance of each candidate entity for an ambiguous mention to each of the unambiguous mentions in the same tweet and tweets in a time slice from past.

## [Location Disambiguation in Local Searches Using Gradient Boosted Decision Trees (Industrial Paper)][gbt]
In a nutshell, this work disambiguates the query answer based on Gradient boosting on a training dataset with features: (1) distance from the user's location (2) number of businesses contained in a given location (3) number of search hits for a given location, indicated the popularity of the place.

There is nothing in the method, but the related work gave some good pointers to the previous work.

## A WordNet like dataset for geographical entities
**While reading these papers, I stumbled into a great resource called: GeoWordNet [writeup on it][geowordnet-paper]; Which is pretty much like WordNet, but for Geographical entities.**

## [Map-based vs. Knowledge-based Toponym Disambiguation][spatial08]
This short paper from 2008 is the oldest ones I found that used spatial proximity to disambiguate.
They tried two approaches one based on measuring the distances on the map and other on finding the disambiguations that finds dense clusters in the GeoWordNet like resource.

Map-based approaches finds a mean over all unambiguous mentions and possible resolutions of all the ambiguous mentions such that no location is beyond the 2$$\sigma$$ distance from the center.
They showed that this method works best when there is enough context i.e. the entire document rather than a sentence or paragraph.
Also, the method failed to perform better than knowledge based method for reasons that are not analyzed in the paper. 

## [Grounding Toponyms in an Italian Local News Corpus][italian-np]
The algorithm in a nutshell

1. Identify the next ambiguous toponym t with senses S = $$(s_1, \cdots , s_n)$$
2. Find all toponyms tc in context;
3. Add to the context all senses C = $$(c_1, \cdots , c_m)$$ of the toponyms in context (if a context toponym has been already disambiguated, add to C only that sense);
4. $$\forall c_i \in C, \forall s_j \in S$$ calculate the map distance $$d_M(c_i, s_j)$$ and text distance $$d_T(c_i, s_j)$$;
5. Combine frequency count $$F(c_i)$$ with distances in order to calculate, for all $$s_j$$ : $$F_i(s_j ) = \sum_{c_i\in C}{\frac{F(c_ii)}{(d_M(c_i,s_j)\cdot d_T (c_i,s_j))^2}}$$ ;
6. Resolve t by assigning it the sense s = $$arg_{s_j\in S} max F_i(s_j)$$.
7. Move to next toponym; if there are no more toponyms:
stop.

[italian-np]: http://dl.acm.org/citation.cfm?id=1722099 "Grounding Toponyms in an Italian Local News Corpus"
[spatial08]: http://dl.acm.org/citation.cfm?id=1460011 "Map-based vs. Knowledge-based Toponym Disambiguation"
[geowordnet-paper]: http://livingknowledge.europarchive.org/images/publications/GeoWordNet.pdf "GeoWordNet: a resource for geo-spatial applications"
[gbt]: http://www.research.att.com/export/sites/att_labs/techdocs/TD_100100.pdf "Location Disambiguation in Local Searches Using Gradient Boosted Decision Trees (Industrial Paper)"
[socialnlp15]: http://www.aclweb.org/anthology/W15-1701 "Location Name Disambiguation Exploiting Spatial Proximity and Temporal Consistency"
[geocode-survey]: http://onlinelibrary.wiley.com/doi/10.1111/tgis.12212/abstract "Automated Geocoding of textual documents: A survey of current approaches"
[emnlp11]: http://www.aclweb.org/anthology/D11-1074 "Linking Entities to a Knowledge Base with Query Expansion"
[shrotri-report]: https://drive.google.com/file/d/0B7-JQZLBkT1AVnQ0dHdYeHN5QU9pOFlPUWJzZ1Q3b290UWNB/view?usp=sharing "Application of Click Logs in Geosensitive Query Processing"
[tkde17-clicklogs]: http://ieeexplore.ieee.org/document/7922603/ "Large-scale Location Prediction for Web Pages"
[gir04]: http://dl.acm.org/citation.cfm?id=1096991 "Detecting geographic locations from web resources"
[vldb00]: http://www.cs.columbia.edu/~gravano/Papers/2000/vldb00.pdf "Computing Geographical Scopes of Web Resources"
[webawhere]: http://dl.acm.org/citation.cfm?id=1009040 "Web-a-Where: Geotagging Web Content"
