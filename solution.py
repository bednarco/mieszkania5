import torch
import pandas

n = 19900
learn_rate = torch.tensor(0.00001)

A = torch.randn((1,3), requires_grad=True)
b = torch.randn(1,requires_grad=True)

def normalize(data):
    data_mean = torch.mean(data, dim=0)
    data_max = torch.max(data, dim=0)[0]
    data_min = torch.min(data, dim=0)[0]
    data = (data-data_mean)/(data_max-data_min)
    return data


def model(x):
    return A.mm(x) + b

def loss(y_predicted, y_original):
    return torch.sum((y_predicted - y_original) ** 2)


data = pandas.read_csv('train/train.tsv', sep='\t',header=None)

#clean data
data[8] = [float(str(val).replace(' ','').replace(',','.')) for val in data[8].values]
data[6] = [float(str(val).replace('więcej niż 10','11')) for val in data[6].values]

x1 = normalize(torch.tensor(data[8], dtype=torch.float)) #Metraż
x2 = normalize(torch.tensor(data[6], dtype=torch.float)) #Liczba pokoi
y = torch.tensor(data[0], dtype=torch.float)


x = torch.stack((x1,x2,torch.ones(x1.size())),0)

for i in range(n):
    ypredicted = model(x)
    cost = loss(ypredicted, y)
    print(A," ", b ," => ", cost)
    cost.backward()

    with torch.no_grad():
        A = A - learn_rate * A.grad
        b = b - learn_rate * b.grad

    A.requires_grad_(True)
    b.requires_grad_(True)


dev_data = pandas.read_csv('dev-0/in.tsv', sep='\t',header=None)
dev_data[7] = [float(str(val).replace(' ','').replace(',','.')) for val in dev_data[7].values]
dev_data[5] = [float(str(val).replace('więcej niż 10','11')) for val in dev_data[5].values]


dev_input1 = normalize(torch.tensor(dev_data[7], dtype=torch.float))
dev_input2 = normalize(torch.tensor(dev_data[5], dtype=torch.float))
dev_x = torch.stack((dev_input1,dev_input2,torch.ones(dev_input1.size())),0)


dev_out = model(dev_x)


dev_file = open("dev-0/out.tsv","w+")
for i in range(0,462):
    dev_file.write(str(dev_out[:,i].data.item()) + "\n")
dev_file.close()


test_data = pandas.read_csv('test-A/in.tsv', sep='\t',header=None)
test_data[7] = [float(str(val).replace(' ','').replace(',','.')) for val in test_data[7].values]
test_data[5] = [float(str(val).replace('więcej niż 10','11')) for val in test_data[5].values]

test_input1 = normalize(torch.tensor(test_data[7], dtype=torch.float))
test_input2 = normalize(torch.tensor(test_data[5], dtype=torch.float))
test_x = torch.stack((test_input1,test_input2,torch.ones(test_input1.size())),0)

test_out = model(test_x)

test_file = open("test-A/out.tsv","w+")
for i in range(0,418):
    test_file.write(str(test_out[:,i].data.item()) + "\n")
test_file.close()