# ----- import module ----- #
import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# ----- Hyper parameter ----- #
CorrThreshold = 0.7  # Threshold for highly correlated pairs
L = 5
startIndex = 0



# ----- Read files ----- #
# change directory
os.chdir(r"C:\....\Predict_COVID")  # alter the directory !!!

# Open CSV file
with open('covid_19.csv', 'r') as csvFile:
    reader = csv.reader(csvFile, delimiter=',')
    dataTemp = np.array(list(reader))
    dataTemp = np.delete(dataTemp, [0, 1, 2], 0)  # Delete first 3 rows
    dataTemp = np.delete(dataTemp, [1, 2], 1)  # Delete 1, 2 cols

# Turn ndarray to dictionary
CSVdict = {}
for i in range(len(dataTemp)):
    CSVdict.update( {dataTemp[i][0] : list(map(float,dataTemp[i][1:-1])) } )

# Get Sequence 'Seq'
Seqdict = {}
for country in CSVdict:
    Seq = np.array(CSVdict[country][1:]) - np.array(CSVdict[country][:-1])  # difference sequence
    Seqdict.update( { country : Seq } )  # {"country": Diff_ndarray}



# ----- Find correlated countries ----- #
# Compute correlation coefficient (CC)
Corr = []
C = set()  # store high-correlated countries 
for country1 in Seqdict:
    CorrVec = []
    for country2 in Seqdict:
        corrValue = np.corrcoef(Seqdict[country1], Seqdict[country2])
        CorrVec.append(corrValue[0, 1])
        
        # add this pair of countries if they are highly correlated
        if (country1 != country2)and(corrValue[0, 1] >= CorrThreshold):
            C.add(country1)
            C.add(country2)

    Corr.append(CorrVec)
Corr = np.array(Corr)

# Plot these coefficients in all the pairs
fig, ax = plt.subplots(1, 1)
x, y = np.mgrid[0:185, 0:185]
CorrRev = Corr[::-1, :]
z = CorrRev
mesh = ax.pcolormesh(x, y, z)
fig.colorbar(mesh)
plt.show()



# ----- Generate the pair (input,target) for modelling ----- #
# Preprocess Input & label
TrainData = []
TrainLabel = []
for country in C:
    # generate sequence L for each label
    CountrySeq = Seqdict[country]
    for i in range(len(CountrySeq) - L - startIndex):
        # add input segment
        TrainData.append(CountrySeq[(startIndex + i) : (startIndex + i + L)])
        # add label
        if CountrySeq[(startIndex + i + L)] > CountrySeq[(startIndex + i + L - 1)]:
            TrainLabel.append([0, 1])  # increase
        else:
            TrainLabel.append([1, 0])  # decrease

# Transfer to ndarray
InputData = np.array(TrainData) 
Label = np.array(TrainLabel)    

np.save("./InputData.npy", InputData)
np.save("./Label.npy", Label)



# ----- save info for Global Map drawing ----- #
# produce Map label
Map_InputData = []
Map_country = []
for country in Seqdict:
    # generate sequence L for each label
    Map_country.append(country)
    CountrySeq = Seqdict[country]
    Map_InputData.append(CountrySeq[-L:])

Map_country = np.array(Map_country)
Map_InputData = np.array(Map_InputData)
np.save("./Map_InputData.npy", Map_InputData)
np.save("./Map_country.npy", Map_country)


"""
# Plot only 6 country for demo
Corr2 = []
PlotCountry = ['Japan', 'Korea, South', 'China', 'France','US','Taiwan*']
for country1 in PlotCountry:
    CorrVec = []
    for country2 in PlotCountry:
        corrValue = np.corrcoef(Seqdict[country1], Seqdict[country2])
        CorrVec.append(corrValue[0, 1])
    Corr2.append(CorrVec)
Corr2 = np.array(Corr2)

# Plot these coefficients in all the pairs
fig, ax = plt.subplots(1, 1)
x, y = np.mgrid[0:7, 0:7]
CorrRev = Corr2[::-1, :]
z = CorrRev
mesh = ax.pcolormesh(x, y, z)
fig.colorbar(mesh)
plt.show()
fig.canvas.draw()

#for label in ax.xaxis.get_xticklabels():
#    label.set_horizontalalignment('right')
for tick in ax.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('right')
labels = PlotCountry
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.show()
"""


