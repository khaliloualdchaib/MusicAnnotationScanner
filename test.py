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
from torchviz import make_dot
import sys
trainingdata = PatchDataset("PatchData/Patches.csv", "PatchData/DataSplit.json", "Training")
validationdata = PatchDataset("PatchData/Patches.csv", "PatchData/DataSplit.json", "Validation")


model = Autoencoder(500,500)
model.load_state_dict(torch.load("autoencoder.pth"))
model.eval()

img_tensor = validationdata[22][0]
trainingdata.ShowImage(img_tensor)


img_tensor = torch.unsqueeze(img_tensor, 0)
img_tensor = img_tensor.permute(0, 3, 1, 2)

output = model(img_tensor)
output = output.squeeze()
output = output.permute(1,2,0)

image_array = output.detach().numpy()
plt.imshow(image_array)
plt.show()

