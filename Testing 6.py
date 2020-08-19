from ResSCANet import *
import pandas as pd

data = pd.read_csv('six channel data/data.csv',index_col=0).values.reshape(-1,6,4001)
label = pd.read_csv('six channel data/label.csv',index_col=0).values.reshape(-1)


test_data_tensor = torch.Tensor(data[1050:])
test_label_tensor = torch.LongTensor(label[1050:])

model = ResSCANet6(6)
# load the pretrained parameters for the newly contructed model
model.load_state_dict(torch.load('six-channel model params.pkl'))

model.eval()
loss_func = nn.CrossEntropyLoss()

model.eval()
test_out = model(test_data_tensor)
test_loss = loss_func(test_out, test_label_tensor)
prediction = torch.max(torch.softmax(test_out, 1), 1)[1]
test_accuracy = (prediction == test_label_tensor).sum().item() / float(len(test_label_tensor))


print("six channel accuracy on test dataset is :",test_accuracy)
print("six channel loss on test dataset is :",test_loss.data.item())