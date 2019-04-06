# MusicMood

# Dataset 

For the purpose of this exercise, we will build models using these datasets:
• https://raw.githubusercontent.com/rasbt/musicmood/master/dataset/training/train_lyrics_1000.csv
• https://github.com/rasbt/musicmood/blob/master/dataset/validation/valid_lyrics_200.csv

This repository contains end-to-end tutorial-like code samples to help solve
text classification problems using machine learning.

## Prerequisites

*   [TensorFlow](https://www.tensorflow.org/)
*   [Scikit-learn](http://scikit-learn.org/stable/)

## Modules

We have one module for each step in the text classification workflow.

1.  *load_vectorize_data* - Functions to load data from four different datasets. For each
    of the dataset we:

    +   Read the required fields (texts and labels).
    +   Do any pre-processing if required. For example, cleaning data (removing stop words, special chars. etc)
    +   Encode 1 - Happy and 0 - Sad
    +   Split the data into training and validation sets.
    +   Shuffle the training data.

     N-gram and sequence vectorization functions.
     
2.  *explore_data* - Helper functions to understand datasets.

3.  *build_and_train_model* - Helper functions to create multi-layer perceptron and
    separable convnet models.

4.  *tune_ngram_model* - Contains example to demonstrate how you can find the best
    hyper-parameter values for your model.
    
5.  *model_metrics* - Accuracy , Loss, Confussion Matrix, F1 Score, ROC Curve.

6.  *app.py* - Python file for web application on FLASK
## CLAAT Document : https://codelabs-preview.appspot.com/?file_id=17fl-qp1hUTN2e9Ahvmqg5yrm-FrhGjyoKIGlUOs1NZU#0
## Web Application Link : http://infinite-depths-56714.herokuapp.com/ 

### User Doc for Application:

Select the country from the dropdown and enter the top number of songs you want to see the mood of. 
The application will take you to next page, where you can see the mood (Happy / Sad) for each song based on the Lyrics
You can also click and View the Lyrics of the song
