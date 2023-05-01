#Deze file is tijdelijk om ML pipeline te testen
from autoencoder import *
from MLPipeline import *
from PatchDataset import *
from torch.utils.data import DataLoader
from MLPipeline import * 
from toTesnor import *

data = PatchDataset("Patches.csv")
batch_size = 16

dataloader = DataLoader(data, batch_size=batch_size)
model = Autoencoder()

loss_fn = torch.nn.MSELoss()
lr= 0.0001

### Set the random seed for reproducible results
torch.manual_seed(0)

optim = torch.optim.Adam(model.parameters(), lr=lr)
# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pipeline = MLPipeline(model, device, loss_fn, optim)
log_dict = pipeline.train_epochs(1,dataloader)
print(log_dict)