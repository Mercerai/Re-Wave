from ResSCANet import *
from CNN1D import *
import pandas as pd
import torch.utils.data as Data


BATCH_SIZE = 64
LR = 1e-4
EPOCH = 3
PRE_TRAINED = False


train_df = pd.read_csv('single channel data/training_data.csv',index_col=0).values
test_df = pd.read_csv('single channel data/testing_data.csv',index_col=0).values

train_values = train_df[:,:-1]
train_labels = train_df[:,-1]

test_values = test_df[:,:-1]
test_labels = test_df[:,-1]

train_data = torch.Tensor(train_values).requires_grad_(True).reshape((-1,1,4001))
train_target = torch.LongTensor(train_labels)

test_data_lst = [torch.Tensor(test_values[:700]).reshape((-1,1,4001)),
              torch.Tensor(test_values[700:1400]).reshape((-1,1,4001)),
              torch.Tensor(test_values[1400:]).reshape((-1,1,4001))]
test_target_lst = [torch.LongTensor(test_labels[:700]),
                torch.LongTensor(test_labels[700:1400]),
                torch.LongTensor(test_labels[1400:])]

dataloader = Data.DataLoader(Data.TensorDataset(train_data,train_target),batch_size = BATCH_SIZE,shuffle=True)

model = ResSCANet(1)
# model = cnn1d(1)
# model = ResNet(1)
if PRE_TRAINED == True:
    model.load_state_dict(torch.load('')) #selectively load pre-trained parameters

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
        # print(h)
        # if epoch >= 2: break
        if i % 2 == 0:
            model.eval()
            # evaluate model on test dataset
            test_accuracy = 0
            test_loss = 0
            for k in range(3):
                
                test_out = model(test_data_lst[k])
                test_loss += loss_func(test_out, test_target_lst[k])
                prediction = torch.max(torch.softmax(test_out, 1), 1)[1]
                test_accuracy += ((prediction == test_target_lst[k]).sum().item() / float(len(test_target_lst[k])))
            test_accuracy /= 3
            test_loss /= 3
            # on this training batch
            # train_out = model(bx)
            # train_prediction = torch.max(torch.softmax(train_out, 1), 1)[1]
            # train_accuracy = ((train_prediction == by).sum().item() / float(BATCH_SIZE))
            model.train()

            print("epoch : ", epoch, " | train loss : ", loss.data.item(), " | test loss : ", test_loss.data.item(),
                   " | test accuracy : ", test_accuracy)

# torch.save(model.state_dict(), 'results.pkl')
