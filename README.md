# Sentiment Analyzer
A Node.Js web service for sentiment analysis. It analyze reviews of hotels and tells if they are `positive` or `negative`.

## Installation & running
To install:
```shell
npm install
./install_python_dependencies.sh
python training.py
```

To run:
```shell
npm start
```

### Docker (recommended)
Build:
```shell
docker build -t sentiment-analyzer .
```
Run:
```shell
docker run --name sent-an -p 3000:3000 sentiment-analyzer
```
Clean image and containers:
```shell
docker stop sent-an
docker rm sent-an
docker rmi sentiment-analyzer
```

## Front-end and back-end
Since I was more familiar with Node.Js and Express.Js for building simple webservers, I decided to stick to it and run a secondary python process used to predicting reviews.

The first strategy was for each HTTP request, spin the python process with some input parameters. This revealed to be really slow, due to the process initialization and the model deserialization (each response took something between 2-3 seconds).

The improved solution was to keep the python process spinning and communicate with it using stdin and stdout. More elegant solutions using pipes or other communication mechanism were available, but this was the simplest and easiest solution for showing this demo.

## Model training and dataset

Two models were trained on two different datasets: one for italian reviews and one for english reviews.  
The italian dataset available at `./model/data/hotel_reviews_it.csv` is a dataset provided by TripAdvisor.  
The english dataset available at `./model/data/hotel_reviews_en.csv` is a dataset provided by Booking.  
Both are zipped in `./model/data/data.zip`

The english dataset provided text reviews splitted in two parts: `Negative_Review` and `Positive_Review`.
The two were joined and a `class` for the full review was determined discretizing the `Review_Score` (if <= 5 the review is `positive`, `negative` otherwise).
Other than that, since the dataset was highly biased towards positive reviews, the positive reviews were sampled in other to match the number of negative reviews. In this way the english dataset was balanced and ready for training.

Each review was lemmatized and a TFIDF was computed on the resulting bag of words.  
The result was a sparse matrix that will be used in the model training step.  
The model was provided by Scikit-Learn. The chosen one was LinearSVC (a linear support vector classifier).
Since the dataset was quite sparse it should have performed good enough.

The two dataset were splitted in a training set (80% of records) and a test set (20% of records). In order to evaluate the performances of the models.

For the italian model the computed F1 score, was around 0.964.  
For the english model the computed F1 score, was around 0.837. In this case there is room of improvement, such as better stopwords filtering and more data exploration, but the short time window available didn't allowed me to improve the model more since I had to care the webpage and the integration between node and python.

The two trained model where persisted to file using the `pickle` native python library. Since I was working on a Windows machine, there's a chance that using linux `pickle` will have same incompatibilities with the persisted files. In this case, just train the models yourself using the training pipeline (1 minute execution):
```shell
python training.py
```

For more information about the processing and training pipeline please see the related notebook at `./model/training.ipynb`.
