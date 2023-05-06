#Deze file is tijdelijk om ML pipeline te testen
from autoencoder import *
from MLPipeline import *
from PatchDataset import *
from torch.utils.data import DataLoader
from MLPipeline import * 
from toTensor import ToTensor
from torchvision import transforms

trainingdata = PatchDataset("Patches.csv", "DataSplit.json", "Training",transform=transforms.Compose([ToTensor()]))
testdata = PatchDataset("Patches.csv", "DataSplit.json", "Testing",transform=transforms.Compose([ToTensor()]))
validationdata = PatchDataset("Patches.csv", "DataSplit.json", "Validation",transform=transforms.Compose([ToTensor()]))
batch_size = 16

training_loader = DataLoader(trainingdata, batch_size=batch_size)
testing_loader = DataLoader(testdata, batch_size=batch_size)
validation_loader = DataLoader(validationdata, batch_size=batch_size)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = Autoencoder()

model.to(device)



loss_fn = torch.nn.MSELoss()
lr= 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)

optim = torch.optim.Adam(model.parameters(), lr=lr)


pipeline = MLPipeline(model, device, loss_fn, optim)
log_dict = pipeline.train_epochs(30,training_loader, validation_loader)
print(log_dict) 