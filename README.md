# Spam-Email-Detection-with-Techno-Hacks
Spam Email Detection using LSTM Neural Network
This repository contains code for building and evaluating a Long Short-Term Memory (LSTM) based neural network for detecting spam emails. The model is trained on a dataset containing labeled spam and ham (non-spam) emails. The code performs text preprocessing, tokenization, and sequence padding before training the LSTM model. The performance of the model is evaluated using various metrics, and visualizations are provided to better understand the data and the model's predictions.

Prerequisites
Make sure you have the following libraries installed before running the code:

pandas
numpy
re
wordcloud
nltk
sklearn
tensorflow
matplotlib
seaborn
You can install these libraries using the following command:

bash
Copy code
pip install pandas numpy re wordcloud nltk scikit-learn tensorflow matplotlib seaborn
Usage
Clone the Repository:

bash
Copy code
git clone <repository-url>
Navigate to the Repository:

bash
Copy code
cd <repository-directory>
Run the Code:

Execute the spam_detection_lstm.py file to run the code.

bash
Copy code
python spam_detection_lstm.py
File Descriptions
spam.csv: The dataset containing labeled spam and ham emails.
spam_detection_lstm.py: The main Python script containing the LSTM model implementation, data preprocessing, training, evaluation, and visualizations.
Output
The code generates the following outputs:

Accuracy: The accuracy of the LSTM model on the test data.
Confusion Matrix: A matrix showing true positive, true negative, false positive, and false negative values.
Classification Report: Precision, recall, F1-score, and support for each class.
Data Distribution Plot: Pie chart showing the distribution of spam and ham emails.
Word Cloud for Spam Emails: Visualization representing the most common words in spam emails.
Precision-Recall Curve: Plot showing the relationship between precision and recall.
ROC Curve: Plot showing the receiver operating characteristic curve.
Confusion Matrix Visualization: Heatmap visualization of the confusion matrix.
Top Words in Ham and Spam Emails: Bar charts displaying the most frequent words in ham and spam emails.
Distribution of Text Lengths: Histograms showing the distribution of text lengths for ham and spam emails.
Acknowledgments
This project uses the spam.csv dataset, which is included in this repository and originates from an unknown source.
