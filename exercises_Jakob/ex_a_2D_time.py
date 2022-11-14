import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


def func(x,t):
    return np.sinc(x) * np.atleast_2d(np.exp(-t*0.2))+1 #+ np.random.normal(0, 0.01, len(x))
    # return np.sinc(x) * np.tan(t) #+ np.random.normal(0, 0.01, len(x))


function_to_learn = func

EPOCHS = 200
N = 100
Nt = 120
n = 1
n_batches = 1
batch_size = Nt//n_batches
min_val = -np.pi*2
max_val = np.pi*2
time = [0, 5]

x_data = np.tile(np.linspace(min_val, max_val, N, dtype=np.float32),(Nt, 1)) # use torch.from_numpy
t_data = np.atleast_2d(np.linspace(*time, Nt, dtype=np.float32)).T
# mesh_x, mesh_y = np.asarray(np.meshgrid(x_data,y_data))
# mesh = [mesh_x, mesh_y ]
# mesh_torch = np.dstack((mesh_x, mesh_y ))

# x_extrapolate = np.linspace( 2*np.pi* 3, 2*np.pi* 6, N, dtype=np.float32) # use torch.from_numpy

z_data = function_to_learn(x_data, t_data)
z_data = z_data.astype(np.float32).T
# plt.imshow(z_data)
# plt.show()

# print(mesh_x.flatten())

train_set = TensorDataset(torch.from_numpy(x_data), torch.from_numpy(z_data.T), torch.from_numpy(t_data))

def collate_fn(data):
    x, z, t = zip(*data)

    x = torch.stack(x)
    z = torch.stack(z).T 
    t = torch.stack(t)
    xt = torch.concatenate([x,t], dim=1)
    
    out = xt.ravel(), z
    return out


train_loader = DataLoader(train_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, drop_last=True)




class NeuralNetwork(nn.Module):
    def __init__(self) :
        super(NeuralNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            # nn.Flatten(),
            nn.Linear((N+1)*batch_size, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 1200),
            nn.LeakyReLU(),
            nn.Linear(1200, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 100*batch_size),
        )


    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits.reshape(N,-1)


# create network
model = NeuralNetwork()
print(model)


# test network on random input value
test_data = np.random.rand((N+1)*batch_size).astype(np.float32)
# test_data = train_loader.next()
print(model.forward(torch.from_numpy(test_data)))



# train network to desired function



# predict function before training
y_prediction = []
for x, _ in train_loader:
    out = model.forward(x).data.numpy()
    y_prediction.append(model.forward(x).data.numpy())

y_prediction = np.hstack(y_prediction)


loss_func = nn.MSELoss() # use a correct loss ie. L2
# loss_func = nn.L1Loss() # use a correct loss ie. L2
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# optimizer = optim.SGD( model.parameters(), lr= 1e-2)

e = 0


losses = []

fig = plt.figure("Training")
ax_im = fig.add_subplot(121)
# line_model  = ax.scatter(x_train, y_prediction.flatten(), label="prediction")
# line_data  = ax.scatter(x_data, y_data, label="data")
im_model  = ax_im.imshow(y_prediction.reshape(N,-1), label="data")
# fig.legend()
# fig.show()
ax_loss = fig.add_subplot(122)
loss_line, = ax_loss.plot([0,EPOCHS*n_batches], [0,1]) # find initial loss

print("Training Model... ")
for epoch in range(EPOCHS):
    e += 1

    y_pred_training = []
    model.train()
    im_train = np.zeros_like(z_data)
    for i, (inputs, targets) in enumerate(train_loader): # create a Dataloader

        # x = torch.from_numpy(np.atleast_1d(data[0])).float()
        # print(x)
        # y = torch.from_numpy(np.atleast_1d(data[1])).float()    
        x = inputs
        y = targets

        
        y_pred = model.forward(x)
        y_pred_training.append(y_pred.data.numpy())
        im_train[:N, (i)*batch_size:(i+1)*batch_size] = y_pred.data.numpy()
        im_model.set_data(im_train)
        im_model.set_clim(np.min(im_train), np.max(im_train))

        # plot_array_model = np.r_[[x.data.numpy(), y_pred.data.numpy()]]
        # plot_array_data = np.r_[[x.data.numpy(), y.data.numpy()]]
        
        # line_model.set_offsets(plot_array_model.T)
        # line_data.set_offsets(plot_array_data.T)
        plt.pause(0.0001)

        optimizer.zero_grad()

        loss = loss_func( y_pred, y)


        loss.backward()

        optimizer.step()

        losses.append(loss.data)
        loss_line.set_xdata(np.arange(len(losses)))
        loss_line.set_ydata(losses)
        ax_loss.set_ylim(-0.1,np.max(losses))

    y_pred_training = np.asarray(y_pred_training)
    
    print(e, end='\r')


y_prediction_trained = []

for x, _ in train_loader:
    y_prediction_trained.append(model.forward(x).data.numpy())

y_prediction_trained = np.hstack(y_prediction_trained)


fig = plt.figure("Function and NN prediction")
axes = fig.subplots(1, 4)
axes[0].imshow( z_data, label="target")
axes[1].imshow( y_prediction, label="untrained")
axes[2].imshow( y_prediction_trained, label="untrained")
ec = axes[3].imshow( z_data - y_prediction_trained.reshape(N, -1), label="untrained")

plt.colorbar(ec, orientation="horizontal")
plt.legend()

plt.figure("Loss")
plt.plot(np.arange(len(losses)), losses)

plt.show()

