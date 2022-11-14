import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


def func(x,y):
    return np.sinc(x)*np.cos(y) #+ np.random.normal(0, 0.01, len(x))


function_to_learn = func


N = 100
n = 1
batch_size = (N//n) **2 * 2
min_val = -np.pi*2
max_val = np.pi*2


x_data = np.linspace(min_val, max_val, N, dtype=np.float32) # use torch.from_numpy
y_data = np.linspace(min_val, max_val, N, dtype=np.float32)
mesh_x, mesh_y = np.asarray(np.meshgrid(x_data,y_data))
mesh = [mesh_x, mesh_y ]
mesh_torch = np.dstack((mesh_x, mesh_y ))

# x_extrapolate = np.linspace( 2*np.pi* 3, 2*np.pi* 6, N, dtype=np.float32) # use torch.from_numpy

z_data = function_to_learn(*mesh)
z_data = z_data.astype(np.float32)

print(mesh_x.flatten())

train_set = TensorDataset(torch.from_numpy(np.c_[mesh_x.flatten(),mesh_y.flatten()]), torch.from_numpy(z_data.flatten()))

def collate_fn(data):
    xy, z = zip(*data)

    xy = torch.stack(xy).ravel()    
    z = torch.stack(z) 
    
    out = xy, z
    return out


train_loader = DataLoader(train_set, collate_fn=collate_fn, batch_size=batch_size//2, shuffle=False, drop_last=True)




class NeuralNetwork(nn.Module):
    def __init__(self) :
        super(NeuralNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(batch_size, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 400),
            nn.LeakyReLU(),
            # nn.Linear(400, 400),
            # nn.LeakyReLU(),
            # nn.Linear(400, 400),
            # nn.LeakyReLU(),
            nn.Linear(400, 200),
            nn.LeakyReLU(),
            nn.Linear(200, batch_size//2),
        )


    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


# create network
model = NeuralNetwork()
print(model)


# test network on random input value
test_data = np.random.rand(batch_size).astype(np.float32)
# test_data = train_loader.next()
print(model.forward(torch.from_numpy(test_data)))



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
EPOCHS = 100

losses = []

fig = plt.figure("Training")
ax = fig.add_subplot()
# line_model  = ax.scatter(x_train, y_prediction.flatten(), label="prediction")
# line_data  = ax.scatter(x_data, y_data, label="data")
im_model  = ax.imshow(y_prediction.reshape(N,-1), label="data")
# fig.legend()
# fig.show()

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

        im_model.set_data(y_pred.data.numpy().reshape(N,-1))
        # plot_array_model = np.r_[[x.data.numpy(), y_pred.data.numpy()]]
        # plot_array_data = np.r_[[x.data.numpy(), y.data.numpy()]]
        
        # line_model.set_offsets(plot_array_model.T)
        # line_data.set_offsets(plot_array_data.T)
        plt.pause(0.0001)

        optimizer.zero_grad()

        loss = loss_func( y_pred, y)

        loss.backward()

        optimizer.step()

        losses.append(loss.data )
    
    print(e, end='\r')


y_prediction_trained = []

for x, _ in train_loader:
    y_prediction_trained.append(model.forward(x).data.numpy())

y_prediction_trained = np.asarray(y_prediction_trained)


fig = plt.figure("Function and NN prediction")
axes = fig.subplots(1, 3)
c = axes[1].imshow( y_prediction.reshape(N, -1), label="untrained")
axes[0].imshow( z_data, label="target")
axes[2].imshow( y_prediction_trained.reshape(N, -1), label="untrained")

plt.colorbar(c, orientation="horizontal")
plt.legend()

plt.figure("Loss")
plt.plot(np.arange(len(losses)), losses)

plt.show()

