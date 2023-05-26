from custom_datasets.FAAMDataset import *
from patches import *
from custom_datasets.PatchDataset import *
from dataTransformation.normalize import Normalize
from dataTransformation.toTensor import ToTensor
from ML.autoencoder import *
from ML.MLPipeline import *
from custom_datasets.PatchDataset import *
from torch.utils.data import DataLoader
from ML.MLPipeline import * 
from torchvision import transforms
from datasplit import Datasplitting
import sys


#-------------------------------------------------- DATA PREP -------------------------------------------------------------------------
#Put here the path to root folder
root = os.getcwd()
dataset = FAAMDataset("jsonfiles/new.json")
patch_width = 500
patch_height = 500
def dataPrep():
    p = Patch(dataset, patch_height, patch_width, root)
    p.CreatePatches()
    Datasplitting(0.60,0.20,0.20,dataset, patch_height=patch_height, patch_width=patch_width)
    n = Normalize("PatchData/DataSplit.json")
    n.normalize_images()
print("Data prep finished")

#-------------------------------------------------- Model Training -------------------------------------------------------------------------

##################### DATA LOADING #####################################
trainingdata = PatchDataset("PatchData/Patches.csv", "PatchData/DataSplit.json", "Training")
testdata = PatchDataset("PatchData/Patches.csv", "PatchData/DataSplit.json", "Testing")
validationdata = PatchDataset("PatchData/Patches.csv", "PatchData/DataSplit.json", "Validation")
batch_size = 16
training_loader = DataLoader(trainingdata, batch_size=batch_size)
testing_loader = DataLoader(testdata, batch_size=batch_size)
validation_loader = DataLoader(validationdata, batch_size=batch_size)



##################### PARAMETERS #####################################
# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = Autoencoder(500, 500)

model.to(device)

loss_fn = torch.nn.MSELoss()

lr= 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)

optim = torch.optim.Adam(model.parameters(), lr=lr)

##################### TRAINING + Validation #####################################
pipeline = MLPipeline(model, device, loss_fn, optim)
log_dict = pipeline.train_epochs(70,training_loader, validation_loader,True)