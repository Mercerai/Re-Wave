from ResSCANet import *
import pandas as pd
import torch.utils.data as Data


test_df = pd.read_csv('single channel data/testing_data.csv',index_col=0).values

test_values = test_df[:,:-1]
test_labels = test_df[:,-1]

test_data_lst = [torch.Tensor(test_values[:700]).reshape((-1,1,4001)),
              torch.Tensor(test_values[700:1400]).reshape((-1,1,4001)),
              torch.Tensor(test_values[1400:]).reshape((-1,1,4001))]
test_target_lst = [torch.LongTensor(test_labels[:700]),
                torch.LongTensor(test_labels[700:1400]),
                torch.LongTensor(test_labels[1400:])]

model = ResSCANet(1)
model.load_state_dict(torch.load('one channel parameters.pkl'))
model.eval()
loss_func=nn.CrossEntropyLoss()
# evaluate model on the test dataset

test_accuracy = 0
test_loss = 0
for k in range(3):
    test_out = model(test_data_lst[k])
    test_loss += loss_func(test_out, test_target_lst[k])
    prediction = torch.max(torch.softmax(test_out, 1), 1)[1]
    test_accuracy += ((prediction == test_target_lst[k]).sum().item() / float(len(test_target_lst[k])))
test_accuracy /= 3
test_loss /= 3

print('single-channel model : test loss:',test_loss,"||","test accuracy:",test_accuracy)