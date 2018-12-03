from os import listdir
from os.path import isfile, join
import unicodedata
import string
import torch
import torch.nn as nn
from random import shuffle
import time,math

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def oneHotEncode(letter):
	oneHotTensor = torch.zeros(1,n_letters)
	letterIndex = all_letters.find(letter)
	oneHotTensor[0][letterIndex] = 1
	return oneHotTensor

def nameToTensor(name):
	nameTensor = torch.zeros(len(name), 1, n_letters)
	index=0
	for letter in name:
		letterIndex = all_letters.find(letter)
		nameTensor[index][0][letterIndex]=1
		index+=1
	return nameTensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(nameTensor, categoryTensor):
	hidden = rnn.initHidden()
	rnn.zero_grad()

	for i in range(nameTensor.size()[0]):
		output, hidden = rnn.forward(nameTensor[i], hidden)

	loss = criterion(output, categoryTensor)
	loss.backward()

	for p in rnn.parameters():
		p.data.add_(-learning_rate, p.grad.data)

	return output, loss.item()


def evaluate(nameTensor):
    hidden = rnn.initHidden()

    for i in range(nameTensor.size()[0]):
        output, hidden = rnn(nameTensor[i], hidden)

    return output

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

print(all_letters)



dataSetPath = "data/names/"
onlyFileNames = []
for f in listdir(dataSetPath):
	if isfile(join(dataSetPath, f)):
		if f[:2]!="._":
			onlyFileNames.append(f)

dataSetComplete={}
completeFileName=""

for file in onlyFileNames:
	completeFileName = dataSetPath+file
	ofile = open(completeFileName,"r")
	currentFileData = ofile.readlines()
	currentFileData = [unicodeToAscii(x.strip()) for x in currentFileData]
	ofile.close()
	langName = file[:-4]
	dataSetComplete[langName]=currentFileData

n_languages = len(dataSetComplete)
all_categories = [s[:-4] for s in onlyFileNames]



n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_languages)

input = nameToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)



print(categoryFromOutput(output))

criterion = nn.NLLLoss()

learning_rate = 0.005

##Converting the dataset dict to an array
wholeDataSet=[]
for category in dataSetComplete:
	for name in dataSetComplete[category]:
		oneExample = []
		oneExample.append(name)
		oneExample.append(category)
		wholeDataSet.append(oneExample)
shuffle(wholeDataSet)
# for i in range(100):
# 	print(wholeDataSet[i])

total_names = len(wholeDataSet)
num_training_examples = int((total_names*4)/5)
print(num_training_examples)
trainDataSet = wholeDataSet[:num_training_examples]
testDataSet = wholeDataSet[num_training_examples:]

num_epochs=5


start = time.time()



allLosses=[]

for epoch in range(num_epochs):
	currentLoss = 0
	i=0
	lossesRecord=[]
	totalLoss=0
	shuffle(trainDataSet)
	for example in trainDataSet:
		categoryTensor = torch.tensor([all_categories.index(example[1])], dtype=torch.long)
		nameTensor = nameToTensor(example[0])
		output, loss = train(nameTensor, categoryTensor)
		currentLoss+=loss
		totalLoss+=loss
		i+=1
		if i%2000==0:
			print('Epoch : %d Iteration: %d Time: %s CurrentLoss: %.4f TotalLoss: %.4f ' % 
				(epoch+1,i,timeSince(start), loss,totalLoss))
		if i%2000==0:
			lossesRecord.append(currentLoss/2000)
			currentLoss=0
	allLosses.append(lossesRecord)




correct=0
for example in testDataSet:
	nameTensor = nameToTensor(example[0])
	category = example[1]
	output = evaluate(nameTensor)
	guess, guess_i = categoryFromOutput(output)
	if guess == category:
		correct+=1

total = len(testDataSet)
accuracy = correct*100.0/total
print("Accuracy = "+str(accuracy))










