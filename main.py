from os import listdir
from os.path import isfile, join
import unicodedata
import string
import torch
import torch.nn as nn

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

print(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

'''First step : Loading the dataset'''

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

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_languages)

input = nameToTensor('Albert')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))







