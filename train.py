#Deze file is tijdelijk om ML pipeline te testen
from autoencoder import *
from MLPipeline import *
from PatchDataset import *
from torch.utils.data import DataLoader
from MLPipeline import * 

trainingdata = PatchDataset("Patches.csv", "DataSplit.json", "Training")
testdata = PatchDataset("Patches.csv", "DataSplit.json", "Testing")
validationdata = PatchDataset("Patches.csv", "DataSplit.json", "Validation")
batch_size = 16

training_loader = DataLoader(trainingdata, batch_size=batch_size)
testing_loader = DataLoader(testdata, batch_size=batch_size)
validation_loader = DataLoader(validationdata, batch_size=batch_size)
model = Autoencoder()

loss_fn = torch.nn.MSELoss()
lr= 0.1

### Set the random seed for reproducible results
torch.manual_seed(0)

optim = torch.optim.Adam(model.parameters(), lr=lr)
# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pipeline = MLPipeline(model, device, loss_fn, optim)
log_dict = pipeline.train_epochs(100,training_loader, validation_loader)
print(log_dict)