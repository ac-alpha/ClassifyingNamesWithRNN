# ClassifyingNamesWithRNN

In this project, I have trained a classifier which classifies the names (rather surnames) into their possible origin country.
The classifier is built using a simple character based Recurrent Neural Network (RNN) model.

## Dataset Used

The dataset used for training the classifier can be downloaded from this [link](https://download.pytorch.org/tutorial/data.zip).
The dataset contains over 20,000 names labelled across 18 different languages. The data is present as seperate text files for each language
in which the names belonging to that language are present in different lines.

## Model Used

Each character of the name is input sequentially to the RNN unit along with the previous hidden state. The output of the RNN unit
is the predicted category along with the new hidden state. Actually a
