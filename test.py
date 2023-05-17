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
trainingdata = PatchDataset("PatchData/Patches.csv", "PatchData/DataSplit.json", "Training")
print(trainingdata[0][0].shape)

model = Autoencoder()
model.load_state_dict(torch.load("autoencoder.pth"))
model.eval()
img_tensor = trainingdata[0][0]
print(img_tensor.max())



"""
trainingdata.ShowImage(img_tensor)
img_tensor = torch.unsqueeze(img_tensor, 0)
img_tensor = img_tensor.permute(0, 3, 1, 2)

output = model(img_tensor)
output = output.squeeze()
output = output.permute(1,2,0)

image_array = output.detach().numpy()
plt.imshow(image_array)
plt.show()
"""
