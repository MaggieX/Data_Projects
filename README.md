# Data Science Projects
A sample of projects done as part of the [Data Incubator fellowship](https://www.thedataincubator.com/), an intensive, 8 week, bootcamp-style program that helps top data and science talent leaving academia with advanced degrees transition to roles as professional data scientists.

The projects were done in Jupyter/ iPython notebooks using the Digital Ocean cloud computing platform.

Please feel free to inquire for more detailed code, out of considerations for current and future fellows only a few sample codes are posted.

## (1) Webscraping - Examining the NYC Socialite Network

<a href="http://www.newyorksocialdiary.com/">The New York Social Diary</a> provides a fascinating lens onto New York's socially well-to-do. As shown in <a href="http://www.newyorksocialdiary.com/party-pictures/2014/holiday-dinners-and-doers">this report of a holiday party</a>, almost all the photos have annotated captions labeling their subjects. We can think of this as an implied social graph: there is a connection between two individuals if they appear in a picture together.

Methodologically this was completed by using [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) to scrape  website pages for photo captions up to a date cut off, a total of ~93,000. These captions were then parsed to return unique names, revealing a total of ~110,000 nodes in the social network. The structure of this social network was analyzed via node degrees and pagerank algorithm. From this popularity and influence can be gaged, and strength of connections (spouse, friends, family) can be proxied. 

_TOOLS USED_: Python - BeautifulSoup, regex, networkx, matplotlib

![network graph](https://github.com/cicilishuaili/DS_projects/blob/master/images/Social_graph.jpg)

## (2) Machine Learning - Predicting Yelp Ratings

<a href="https://www.yelp.com/developers/documentation/v2/business">The Yelp dataset</a> contains unstructured meta data about each venue (city, latitude/longitude, category descriptions, etc), and a star rating. Predicting a new venue's popularity from such information makes for a great ML problem. It had all the classics from data wrangling in JSON, feature engineering, creating custom transformer in the ML pipeline, to an [ensemble regressor](http://scikit-learn.org/stable/modules/ensemble.html) 

_TOOLS USED_: numpy, pandas, sklearn

## (3) MapReduce -- Analyzing Wikipedia

A large set of English Wikipedia was scraped to determine its most frequently used words and link statistics for the unique links on each page.

Distributed computing is ideal for these types of tasks, as they allow for parallel processing of large data sets across node clusters. Hadoop Distributed File System (HDFS) is a distributed file system that provides high-throughput access (multi-terabyte data-sets) to application data, a Hadoop YARN is the framework for job scheduling and cluster resource management within the HDFS environment. With these two in place, one can run MapReduce jobs. Utilizing the [MRJob](https://github.com/Yelp/mrjob) Python package developed at Yelp, one can write MapReduce jobs in Python.

_TOOLS USED_: Hadoop MapReduce, MRJob, BeautifulSoup, Google Cloud Platform, AWS

## (4) SQL - Investigating NYC Restaurant Inspections

The city of New York inspect roughly 24,000 restaurants a year and assigns a grade to restaurants after each inspection, over a decade this creates a public dataset of over 500,000 records. SQL was used to parse and analyze a decade worth of NYC Restaurant Inspections data. Different slices determining the grade distribution by zipcode, borough, and cuisine were extracted, with some cuisines tended to have a disproportionate number of which violations.

![cartoDB map](https://github.com/cicilishuaili/DS_projects/blob/master/images/Carto_DB_map.png)

_TOOLS USED_: SQL, CartoDB

## (5) Data Visualization

Interactive visualizations are used over the course of the project. Bokeh, Flask, and Heroku all come together to visually engage, describe and inform. For more details refer to the [Flask Demo App Repo](https://github.com/cicilishuaili/Flask) and [Capstone Project Repo](https://github.com/cicilishuaili/Ax-Gender-Tax).

![Flask demo](https://github.com/cicilishuaili/DS_projects/blob/master/images/Stock.png)

![Capstone](https://github.com/cicilishuaili/DS_projects/blob/master/images/tf-idf.png)

_TOOLS USED_: Flask, Pandas, Heroku, PostGreSQL

## (6) NLP - Predicting Yelp Ratings

Given the richness of information contained in the texts, Yelp review data was explored for in the context of predicting ratings. Various natural language processing (NLP) techniques were explored on the text data. At its most fundamental, the words need to be transformed into quantities via tokenizers and vectorizers.

_TOOLS USED_: nltk, numpy, pandas, sklearn, bigrams, n-grams

## (7) Time Series - Predicting the Weather

Time series prediction presents its own set of unique challenges in ML problems. A linear regression would likely fail on the basis of the existence of autocorrelation. Periodicity/ seasonality and drift add to considerations. Sliding windows and forward chaining need to replace traditional cross-validation techniques. 

Two time series models are built to predict weather in a given city at a given time, based on historical trends. There are two ways to handle seasonality. The simpler (and perhaps more robust) is to have a set of indicator variables. That is, make the assumption that the temperature at any given time is a function of only the month of the year and the hour of the day, and use that to predict the temperature value. A more complex approach is to fit/transform our model. Since we know that temperature is roughly cyclical on an annual and daily basis, we know that a reasonable model might be:

![math eqn](https://latex.codecogs.com/gif.latex?%24%24%20y_t%20%3D%20y_0%20%5Csin%5Cleft%282%5Cpi%5Cfrac%7Bt%20-%20t_0%7D%7BT%7D%5Cright%29%20&plus;%20%5Cepsilon%20%24%24)

where ![alt text](https://latex.codecogs.com/gif.latex?%24k%24) and ![alt text](https://latex.codecogs.com/gif.latex?%24t_0%24) are parameters to be learned and ![alt text](https://latex.codecogs.com/gif.latex?%24T%24) is one year for seasonal variation.  While this is linear in ![alt text](https://latex.codecogs.com/gif.latex?%24y_0%24), it is not linear in ![alt text](https://latex.codecogs.com/gif.latex?%24t_0%24). However, ffrom <a href="https://en.wikipedia.org/wiki/Fourier_analysis">Fourier analysis</a>, the above is equivalent to

![math eqn2](https://latex.codecogs.com/gif.latex?%24%24%20y_t%20%3D%20A%20%5Csin%5Cleft%282%5Cpi%5Cfrac%7Bt%7D%7BT%7D%5Cright%29%20&plus;%20B%20%5Ccos%5Cleft%282%5Cpi%5Cfrac%7Bt%7D%7BT%7D%5Cright%29%20&plus;%20%5Cepsilon%20%24%24)

which is linear in ![alt text](https://latex.codecogs.com/gif.latex?%24A%24) and ![alt text](https://latex.codecogs.com/gif.latex?%24B%24).

_TOOLS USED_: numpy, pandas, sklearn

## (8) Spark - Analyzing StackOverflow Data

[StackOverflow](https://stackoverflow.com/) is a collaboratively edited question-and-answer site focused on programming topics. Because of the variety of features tracked, including a variety of feedback metrics, it allows for some open-ended analysis of user behavior on the site.

StackExchange (the parent organization) provides an anonymized <a href="https://archive.org/details/stackexchange">data dump</a>, this project used Spark for data manipulation, analysis, and machine learning. Similar to the MapReduce project, this is an ideal use for distributed computing. <a href="https://spark.apache.org/">Spark</a> is Hadoop's bigger, better, stronger, faster cousin -- and runs on top of HDFS with the ability to cache, significantly increasing the speed over traditional Hadoop/MapReduce jobs.  

Using [PySpark](https://spark.apache.org/docs/0.9.0/python-programming-guide.html), a massive dataset of unstructured XML files was analyzed. With the size, we can start to train and build word embeddings via Word2Vec to find synonyms.

See [here](https://github.com/cicilishuaili/Data_Projects/blob/master/spark-edited.ipynb) for a version of the notebook (edited for brevity).

_TOOLS USED_: Spark, PySpark, Spark MLlib, Word2Vec


## (9) TensorFlow - Neural Networks for Image Classification

Neural networks are all the rage in ML, and deservedly so for their high performance in tasks that spans far beyond image classification. In this project, a series of models are built to classify a series of images into one of ten classes ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'). For expediency, these images are pretty small (32×32×3). This can make classification a bit tricky—-human performance is only about 94%. 

![example image](https://github.com/cicilishuaili/DS_projects/blob/master/images/TF_image.png)

The image above is a frog. (Now you see it!)

TensorFlow is a popular framework as it is an open-source software library for "dataflow" programming. Computations are expressed as stateful dataflow graphs. Computations are expressed as stateful dataflow graphs. The name TensorFlow derives from the operations that such neural networks perform on multidimensional data arrays (AKA "tensors"). 

A multi-layer fully-connected neural network achieves an accuracy of about 44% on a training set and 41% on a test set. A simple convolutional neural net achieves accuracy of 80% on a training set and 70% on a test set. A simple transfer learning model based on GoogLeNet achieves a training accuracy of 87% and a test accuracy of 85%.

_TOOLS USED_: TensorFlow

## (Optional) ML to Categorize Music Samples

Audio/music offers another source of rich data. The objective of this miniproject is to develop models that are able to recognize the genre of a musical piece (_electronic, folkcountry, jazz, raphiphop, rock_), first from pre-computed features and then from the raw waveform (input files with 5-10 seconds of a music sample). This is a typical example of a classification problem on time series data. 

_TOOLS USED_: Librosa, sklearn, PCA
