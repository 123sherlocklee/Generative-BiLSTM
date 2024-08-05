# Reverse Method for DGA Algorithm Based on Generative BiLSTM

author: Bowen Li ;
Yanchen Qiao ;
Weizhe Zhang ;
Yu Zhang

DGA (Domain Generation Algorithm) is a technique used to generate a large number of domain names, widely utilized for malware communication. Traditional methods for intercepting DGA domains involve using machine learning to detect whether a domain belongs to DGA, which not only demands high computational resources but also suffers from interception latency. This paper proposes a reverse  method for DGA algorithms based on a generative BiLSTM model. This method uses the BiLSTM model to learn the patterns of DGA domain sequences of a particular type, thereby reversing the DGA algorithm to preemptively generate a blacklist of domains for that type of DGA algorithm. This improves the timeliness and accuracy of domain interception. Experimental results show that the model can effectively reverse multiple types of DGA algorithms and generate subsequent DGA domains that might be produced by these algorithms.

![Uploading image.png…]()

