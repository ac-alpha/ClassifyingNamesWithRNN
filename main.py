from os import listdir
from os.path import isfile, join
import unicodedata

'''First step : Loading the dataset'''

dataSetPath = "data/names/"
onlyFileNames = []
for f in listdir(dataSetPath):
	if isfile(join(dataSetPath, f)):
		if f[:2]!="._":
			onlyFileNames.append(f)
# print(onlyFileNames)			#for debugging

dataSetComplete={}
completeFileName=""

for file in onlyFileNames:
	completeFileName = dataSetPath+file
	ofile = open(completeFileName,"r")
	currentFileData = ofile.readlines()
	currentFileData = [unicodedata.normalize('NFKD', x.strip()).encode('ascii','ignore') for x in currentFileData]
	ofile.close()
	langName = file[:-4]
	dataSetComplete[langName]=currentFileData
# print(dataSetComplete["Czech"])		#for debugging


