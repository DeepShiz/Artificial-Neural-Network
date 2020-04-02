from network import Network
import numpy as np
import random
import pandas as pd

def split_dataset(data):
    actual = data
    data_indices = [i for i in range(len(data))]
    test_indices = random.sample(range(len(data)), int(0.1*len(data)))
    data_indices = np.delete(data_indices, test_indices)

    train = []
    val = []
    test = []
    data = data.values.tolist()
    for i in range(0,len(data)):
        if(i in data_indices):
            train += [data[i]]
        else:
            test += [data[i]]
    train = []
    for i in range(0, len(data_indices)):
        if(i%5==0):
            val += [data[i]]
        else:
            train += [data[i]]

    train = pd.DataFrame(train, columns = actual.columns)
    val = pd.DataFrame(val, columns = actual.columns)
    test = pd.DataFrame(test, columns = actual.columns)
    return train, val, test

def toy_problem():

    data = np.sin(np.linspace(-2*np.pi, 2*np.pi, 1000))

    train_data = [(x,y) for x,y in zip(np.linspace(-2*np.pi, 2*np.pi, 1000),data)]
    print(type(train_data))
    test_data = random.choices(train_data, k = 100)

    network = Network([1,20,4,1])

    network.stoch_gradient_descent(train_data, 1000, 200, 0.01)

    network.output_final(np.linspace(-2*np.pi,2*np.pi,1000), data)

data = pd.read_excel("uci.xlsx")
train, val, test = split_dataset(data)

network = Network([4,20,5,1])
train_data = []     #[](x,y) for x,y in zip(train[["AT","V","AP","RH"]], train[["PE"]])]

for data in train.values.tolist():
    train_data += [(data[0:4], data[-1])]

print(train_data[0])
network.stoch_gradient_descent(train_data, 100, 200, 0.01)
print(network.predict(train[["AT","V","AP","RH"]], train[["PE"]]))


#toy_problem()
