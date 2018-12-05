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

## Code

The actual code is well documented and explained [here](https://github.com/the-bat-hunter/ClassifyingNamesWithRNN/blob/master/name_classifier.ipynb)

## Implementation Details

Negative Log Likelihood loss has been used to compute the losses between predictions and actual outputs. 
Total dataset of 20000 images was divided into training(80%) and testing(20%) sets. The model was trained for 5 epochs. and the weights were updated using gradient descent algorithm. 

Total time of running the whole script was 1 min 55 sec on an i5 5th gen CPU.

## Results and Observations

1. The accuracy on the dataset was 73% approximately which is acceptable seeing the size of the dataset.
2. Adding extra epochs does not improve the result very much which is observable from the graph of losses of all the five epochs. The losses seem to diminish in successive iterations only in the first epoch. In rest of the epochs the losses kept variating randomly but the loss decreased overall in an epoch.
3. Although it is recommended that the training to test data ration should be 80:20 but by varying this ratio to 90:10, the results actually got improved although very slightly(close to 1 %) but it only leaves some 2000 examples for testing. Whereas with a ratio of 70:30, the results got degraded by more than one percent. So, I have used 80:20 ratio. The results of 70:30 and 90:10 are available on different commits.
4. Even in the 4th epoch or 5th epoch,  there are some examples which have very high loss. This shows that although the average loss is decreasing per epoch/iteration, but there may be some such examples present where the model may predict extremely different results from reality.

## License
See [LICENSE](https://github.com/the-bat-hunter/ClassifyingNamesWithRNN/blob/readme/LICENSE)
