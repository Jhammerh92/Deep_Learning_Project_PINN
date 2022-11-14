import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


def func(x, diff=False):
    return -np.cos(x) if diff else np.sin(x) #+ np.random.normal(0, 0.01, len(x))



function_to_learn = func


N = 100
n = 1
n_batches = 1
batch_size = N//n_batches #N//1
min_val = -np.pi*2
max_val = np.pi*2



x_data = np.linspace(min_val, max_val, N, dtype=np.float32) # use torch.from_numpy
x_train = x_data[::n]
x_extrapolate = np.linspace( 2*np.pi* 3, 2*np.pi* 6, N, dtype=np.float32) # use torch.from_numpy
y_data = function_to_learn(x_data)
y_data = y_data.astype(np.float32)
y_train = y_data[::n]

train_set = TensorDataset(torch.tensor(x_train, requires_grad=True), torch.tensor(y_train, requires_grad=True))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True)




class NeuralNetwork(nn.Module):
    def __init__(self) :
        super(NeuralNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(batch_size, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 400),
            nn.LeakyReLU(),
            # nn.Linear(400, 400),
            # nn.LeakyReLU(),
            # nn.Linear(400, 400),
            # nn.LeakyReLU(),
            nn.Linear(400, 200),
            nn.LeakyReLU(),
            nn.Linear(200, batch_size),
        )


    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


# create network
model = NeuralNetwork()
print(model)


# test network on random input value
print(model.forward(torch.rand(batch_size)))



# train network to desired function



# predict function before training
y_prediction = []
for x, _ in train_loader:
    y_prediction.append(model.forward(x).data.numpy())

y_prediction = np.asarray(y_prediction)


loss_func = nn.MSELoss() # use a correct loss ie. L2
# loss_func = nn.L1Loss() # use a correct loss ie. L2
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# optimizer = optim.SGD( model.parameters(), lr= 1e-2)

e = 0
EPOCHS = 150

losses = []

fig = plt.figure("Training")
ax = fig.add_subplot()
line_model  = ax.scatter(x_train, y_prediction.flatten(), label="prediction")
line_data  = ax.scatter(x_data, y_data, label="data")
line_grad  = ax.scatter(x_data, y_data, label="grad")
fig.legend()
fig.show()

for param in model.parameters():
    param.retain_grad()



print("Training Model... ")
for epoch in range(EPOCHS):
    e += 1

    model.train()
    for inputs, targets in train_loader: # create a Dataloader

        # x = torch.from_numpy(np.atleast_1d(data[0])).float()
        # print(x)
        # y = torch.from_numpy(np.atleast_1d(data[1])).float()    
        x = inputs
        y = targets


        y_pred = model.forward(x)
        y_pred.retain_grad()
        # dy_pred = y_pred.grad

        y_pred.backward(x, retain_graph=True)
        # print(y_pred.grad)

        plot_array_model = np.r_[[x.data.numpy(), y_pred.data.numpy()]]
        plot_array_data = np.r_[[x.data.numpy(), y.data.numpy()]]
        plot_array_grad = np.r_[[x.data.numpy(), y_pred.grad.data.numpy()]]

        
        line_model.set_offsets(plot_array_model.T)
        line_data.set_offsets(plot_array_data.T)
        line_grad.set_offsets(plot_array_data.T)
        plt.pause(0.0001)

        optimizer.zero_grad()

        loss = loss_func( y_pred, y)
        # loss_grad = loss_func()

        loss.backward()

        optimizer.step()

        losses.append(loss.data )
    
    print(e, end='\r')


y_prediction_trained = []

for x, _ in train_loader:
    y_prediction_trained.append(model.forward(x).data.numpy())

y_prediction_trained = np.asarray(y_prediction_trained)


plt.figure("Function and NN prediction")
plt.plot(x_data, y_data, label="target")
plt.plot(x_train, y_prediction.flatten(), label="untrained")
plt.plot(x_train, y_prediction_trained.flatten(), label="trained")
plt.legend()

plt.figure("Loss")
plt.plot(np.arange(len(losses)), losses)

plt.show()

