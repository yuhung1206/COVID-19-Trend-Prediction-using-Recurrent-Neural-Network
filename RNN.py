# ----- import module ----- #
import torch
from torch import nn
import torch.nn.functional as Func
import torch.utils.data as Data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os



# ----- Load Data ----- #
# change directory
os.chdir(r"C:\....\Predict_COVID")  # alter the directory !!!

InputDataFile = "InputData.npy"
LabelFile = "Label.npy"
MapFile = "Map_InputData.npy"

# load Data & label
AllLabel = np.load(LabelFile)
AllData = np.load(InputDataFile) #DataSize
(DataSize, SeqLen) = AllData.shape
AllData = np.reshape(AllData, (DataSize, SeqLen, 1))  # [DataSize, SeqLen, Input_Size] = [6955, 15, 1]

# Select train data & test data
TrainSize = int(DataSize*7/10.0)  # TrainSize = 5564
TestSize = DataSize - TrainSize  # TrainSize = 1391
Shuff_Ind = np.arange(DataSize)
np.random.shuffle(Shuff_Ind)
Train_Ind = Shuff_Ind[:TrainSize]
TrainData = AllData[Train_Ind]
TrainLabel = AllLabel[Train_Ind]
Test_Ind = Shuff_Ind[TrainSize:]
TestData = AllData[Test_Ind]
TestLabel = AllLabel[Test_Ind]

# load Map data
MapData = np.load(MapFile)

# Hyper Parameters
EPOCH = 200
BATCH_SIZE = TrainSize
TIME_STEP = AllData.shape[1]      # rnn time step
INPUT_SIZE = AllData.shape[2]     # rnn input size (time series = 1)
Hidden_Size = 100
RNN_Layer = 2
LR = 0.01               # learning rate



# ----- Create Dataset ----- #
class TorchDataset(Data.Dataset):
    def __init__(self, Country_Data, Country_Label):
        """
        :param Label: increase(1), decrease(0)
        :param len: sequence length
        """
        self.Label = Country_Label
        self.len = len(self.Label)
        self.Data = Country_Data

        '''class torchvision.transforms.ToTensor'''
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, i):
        index = i % self.len
        label = self.Label[index, :]  # label --> [1 X 2]
        # obtain maximum to normalize # Data  --> [15 X 1]
        # transform from ndarray to torch
        Input = torch.from_numpy(self.Data[index, :])  # Need normalize or not???
        # transform from ndarray to torch
        Input = Input.type(torch.FloatTensor)
        return Input, label

    def __len__(self):
        return self.len



# Data Loader for easy mini-batch return in training [batch=64, TimeStep=15, inputSize=1]
train_data = TorchDataset(TrainData, TrainLabel)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = TorchDataset(TestData, TestLabel)
test_loader = Data.DataLoader(dataset=test_data, batch_size=TrainSize, shuffle=False) #len(test_data)



# ----- Construct Model ----- #
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(         # if use nn.RNN() or nn.LSTM()
            input_size=INPUT_SIZE,
            hidden_size=Hidden_Size,         # rnn hidden unit
            num_layers=RNN_Layer,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1st dimension. e.g. (batch, time_step, input_size)
            dropout=0.3,
        )

        self.out = nn.Linear(Hidden_Size, 2)  # Output: Increase or Decrease

    def forward(self, x):
        r_out, h_state = self.rnn(x, None)

        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)
# ,  betas=(0.9, 0.999), eps=1e-08,weight_decay=0.002
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR,  betas=(0.9, 0.999), eps=1e-08, weight_decay=0.004)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()



# ----- Training and Testing ----- #
h_state = None      # for initial hidden state
TrainAcc = []
TestAcc = []
TrainLoss = []
TestLoss = []
TrainPredict = []

# decomment below section to use the GPU!!!
# device = torch.device("cuda")
# rnn = rnn.to(device)

for epoch in range(EPOCH):
    # ----- Training ----- #
    train_acc = 0.
    train_loss = 0.
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
        b_x = b_x.view(-1, TIME_STEP, INPUT_SIZE)  # reshape x to (batch, seq_length, input_size)
        dataSize = b_x.shape[0]
        prediction = rnn(b_x)
        B_y = torch.max(b_y.view(-1, 2), 1)[1]
        Pred1 = torch.max(prediction, 1)[1].data.numpy().squeeze()
        loss = loss_func(prediction, B_y)

        optimizer.zero_grad()                           
        loss.backward()   # apply backpropagation
        optimizer.step()                                

        num_correct = float((Pred1 == B_y.numpy()).sum())
        train_acc += num_correct
        train_loss += loss.data


    # ----- Testing ----- #
    rnn.eval()
    test_loss = 0.
    test_acc = 0.
    for i, (t_x, t_y) in enumerate(test_loader):
        t_x = t_x.view(-1, TIME_STEP, INPUT_SIZE)  # reshape x to (len(test_data), seq_length, input_size)
        dataSize2 = t_x.shape[0]

        testPrediction = rnn(t_x)

        T_y = torch.max(t_y.view(-1, 2), 1)[1]
        #Test_Pred = testPrediction[:,-1,:]
        Pred2 = torch.max(testPrediction, 1)[1].data.numpy().squeeze()
        loss2 = loss_func(testPrediction, T_y)
        test_loss += loss2.data
        accuracy2 = float((Pred2 == T_y.numpy()).sum())
        test_acc += float(accuracy2)

    print("-----Epoch"+str(epoch)+"-----")
    print('Train Acc: {:.6f}'.format(train_acc/( TrainSize )))
    print('Train Loss: {:.6f}'.format(train_loss/( TrainSize )))
    print('Test Acc: {:.6f}'.format(test_acc/(TestSize)))
    print('Test Loss: {:.6f}'.format(test_loss/(TestSize)))

    # store results in this itertion
    TrainAcc.append(train_acc/( TrainSize ))
    TestAcc.append(test_acc / (TestSize))
    TrainLoss.append(train_loss / (TrainSize))
    TestLoss.append(test_loss / (TestSize))



# ----- Save the performance ----- #
np.save("./TrainAcc.npy", TrainAcc)
np.save("./TestAcc.npy", TestAcc)
np.save("./TrainLoss.npy", TrainLoss)
np.save("./TestLoss.npy", TestLoss)



# ----- Plot the performance ----- #
plt.figure(figsize=(12,5))  
plt.subplot(1, 2, 1)
plt.plot(TrainAcc, color="green", linewidth=1, label="$TrainAcc$")
plt.plot(TestAcc, color="red", linewidth=1, label="$TestAcc$")
plt.xlabel("Epoch(s)")
plt.ylabel("Accuracy")
plt.title("\nAccuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(TrainLoss, color="green", linewidth=1, label="$TrainLoss$")
plt.plot(TestLoss, color="red", linewidth=1, label="$TestLoss$")
plt.xlabel("Epoch(s)")
plt.ylabel("Loss")
plt.title("\nLearning curve")
plt.legend()
plt.show()



# ----- Preparation for World Map ----- #

# function for World Map drawing
def softmax(x):
    # softmax function for vector x
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

# transform from ndarray to torch
Input_Map = torch.from_numpy(MapData)
m_x = Input_Map.type(torch.FloatTensor)
m_x = m_x.view(-1, TIME_STEP, INPUT_SIZE)  # reshape
AllPrediction = rnn(m_x)  # feed the data to trained model
AllPrediction = AllPrediction.data.numpy().squeeze()
# load map information
CountryLabel = np.load("Map_country.npy")
CodeLabel = np.load("CodeArray.npy") 

MapProb = []
IncDict = {}
DecDict = {}

for k in range(185):
    CountryName = CountryLabel[k]
    # activation function: softmax
    tmpProb = softmax(AllPrediction[k,:])
    (r, c) = np.where(CodeLabel == CountryName)
    if (CodeLabel[r,1] != None)and(tmpProb[1] >= tmpProb[0]): # increase
        IncDict.update( { CodeLabel[r,1].item() : tmpProb[1] })
    elif (CodeLabel[r,1] != None)and(tmpProb[1] < tmpProb[0]): # decrease
        DecDict.update({ CodeLabel[r,1].item() : tmpProb[0] })

    MapProb.append(tmpProb)

MapProb = np.array(MapProb)

# save dictionary for World map
np.save("DecDict.npy", DecDict)
np.save("IncDict.npy", IncDict)

