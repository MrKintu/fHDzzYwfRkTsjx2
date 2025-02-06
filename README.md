# fHDzzYwfRkTsjx2

## Overview
This repository contains four distinct models developed as part of the AIDI 2000 course assignment. Each model employs different techniques in natural language processing (NLP) and machine learning to classify text data, specifically focusing on movie reviews. The goal is to predict the sentiment of the reviews, categorizing them into positive or negative classes.

## Models

### 1. Transfer Learning Model
- **Description**: This model utilizes transfer learning techniques with Keras to classify text data. It leverages pre-trained embeddings and convolutional neural networks to improve the accuracy of sentiment analysis.
- **Key Steps**:
  - Importing necessary libraries such as TensorFlow and pandas.
  - Loading and preprocessing the dataset, including handling missing values.
  - Defining a sequential model that includes an embedding layer, convolutional layers, and a dense output layer.
  - Training the model on the training dataset and evaluating its performance on the test dataset.

### 2. Transfer Word Embedding Model
- **Description**: This model implements a sequential neural network that uses word embeddings to represent text data. The embeddings help capture semantic meaning, which is crucial for understanding sentiment.
- **Key Steps**:
  - Importing libraries and loading the dataset.
  - Preprocessing the text data, including tokenization and padding sequences to ensure uniform input size.
  - Defining a model architecture that includes an embedding layer followed by dense layers for classification.
  - Training the model and evaluating its accuracy on the test set, providing insights into its performance.

### 3. Transfer Logistic Regression Model
- **Description**: This model applies logistic regression, a statistical method for binary classification, to predict the sentiment of movie reviews. It utilizes TF-IDF vectorization to convert text data into numerical format.
- **Key Steps**:
  - Importing necessary libraries and loading the dataset.
  - Preprocessing the text data and encoding the labels.
  - Using TF-IDF vectorization to transform the text data into a format suitable for logistic regression.
  - Training the logistic regression model and evaluating its accuracy on the test set, showcasing its effectiveness in sentiment classification.

### 4. Transfer Ensemble Learning Model
- **Description**: This model employs ensemble learning techniques, combining multiple classifiers (Random Forest and Gradient Boosting) to enhance predictive performance. Ensemble methods typically yield better accuracy by leveraging the strengths of different models.
- **Key Steps**:
  - Importing libraries and loading the dataset.
  - Preprocessing the text data and splitting it into training and testing sets.
  - Defining base models using Random Forest and Gradient Boosting classifiers, and creating pipelines for streamlined processing.
  - Training the ensemble models and evaluating their performance, including visualizing confusion matrices to assess classification results.

## Requirements
To run the models, ensure you have the following dependencies installed:
- Python 3.x
- TensorFlow (for deep learning models)
- scikit-learn (for traditional machine learning models)
- pandas (for data manipulation)
- matplotlib (for plotting graphs)
- seaborn (for enhanced visualizations)

## Usage
To execute the models, follow these steps:
1. Clone this repository to your local machine.
2. Navigate to the directory containing the Jupyter notebooks.
3. Open each notebook in a Jupyter environment and run the cells sequentially to train and evaluate the models.

## Conclusion
These models illustrate various approaches to text classification and natural language processing, demonstrating the application of machine learning techniques in real-world scenarios. By analysing movie reviews, we can gain insights into public sentiment, which can be valuable for various applications in the film industry and beyond.
