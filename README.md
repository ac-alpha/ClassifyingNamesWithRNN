# ClassifyingNamesWithRNN

In this project, I have trained a classifier which classifies the names (rather surnames) into their possible origin country.
The classifier is built using a simple character based Recurrent Neural Network (RNN) model.

## Dataset Used

The dataset used for training the classifier can be downloaded from this [link](https://download.pytorch.org/tutorial/data.zip).
The dataset contains over 20,000 names labelled across 18 different languages. The data is present as seperate text files for each language
in which the names belonging to that language are present in different lines.

## Model Used

Each character of the name is input sequentially to the RNN unit along with the previous hidden state. The output of the RNN 
unit is the predicted category along with the new hidden state. Actually the main purpose of the hidden state of any RNN unit is to store some information about the sequence before the current character. 

The model used in training the classifier is taken from [this tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html). 

<p align="center">
<img src="model.png" /></div>

## License
See [LICENSE](https://github.com/the-bat-hunter/ClassifyingNamesWithRNN/blob/readme/LICENSE)
