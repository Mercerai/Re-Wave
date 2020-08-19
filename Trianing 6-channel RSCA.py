from ResSCANet import *
import torch
import random
import torch.utils.data as Data
import pandas as pd
import numpy as np

BATCH_SIZE = 32
LR = 1e-4
EPOCH = 6
PRE_TRAINED = False

data = pd.read_csv('six channel data/data.csv',index_col=0).values.reshape(-1,6,4001)
label = pd.read_csv('six channel data/label.csv',index_col=0).values.reshape(-1)

train_data_tensor = torch.Tensor(data[:1050])
train_label_tensor = torch.LongTensor(label[:1050])
print(train_data_tensor.shape)
test_data_tensor = torch.Tensor(data[1050:])
test_label_tensor = torch.LongTensor(label[1050:])

dataloader = Data.DataLoader(Data.TensorDataset(train_data_tensor,train_label_tensor), batch_size=BATCH_SIZE,shuffle=True)
model = ResSCANet6(6)
if PRE_TRAINED == True:
    model.load_state_dict(torch.load(''))

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# optimizer = torch.optim.RMSProp(model.parameters(), lr=LR)
# optimizer = torch.optim.SGD(model.parameters(), lr=LR)

for epoch in range(EPOCH):
    for i, (bx, by) in enumerate(dataloader):
        out = model(bx)
        optimizer.zero_grad()
        loss = loss_func(out, by)
        loss.backward()
        optimizer.step()
        print('here')
        if i % 2 == 0:
            # evaluate model on test dataset
            model.eval()
            test_out = model(test_data_tensor)
            test_loss = loss_func(test_out, test_label_tensor)
            prediction = torch.max(torch.softmax(test_out, 1), 1)[1]
            test_accuracy = (prediction == test_label_tensor).sum().item() / float(len(test_label_tensor))
            # on this training batch
            # train_out = model(bx)
            # train_prediction = torch.max(torch.softmax(train_out, 1), 1)[1]
            # train_accuracy = ((train_prediction == by).sum().item() / float(BATCH_SIZE))
            model.train()

            print("epoch : ", epoch, " | train loss : ", loss.data.item(), " | test loss : ", test_loss.data.item(),
                   " | test accuracy : ", test_accuracy)

torch.save(model.state_dict(), 'six-channel model params.pkl')