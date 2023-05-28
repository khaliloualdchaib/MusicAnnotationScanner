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
testingdata = PatchDataset("PatchData/Patches.csv", "PatchData/DataSplit.json", "Testing")

index = None
with open("PatchData/DataSplit.json", "r") as f:
    jsonfile = json.load(f)
    len(jsonfile["Testing"])
    for i in range(len(jsonfile["Testing"])):
        #print(jsonfile["Testing"][int(i)])
        if(jsonfile["Testing"][int(i)] == str(7053)):
            index = int(i)

print(jsonfile["Testing"][index])
for i in range(len(testingdata)):
    print(testingdata[i][1])
    
model = Autoencoder(500,500)
model.load_state_dict(torch.load("autoencoder_reduced_filters.pth"))
model.eval()

img_tensor = testingdata[index][0]
testingdata.ShowImage(img_tensor)


img_tensor = torch.unsqueeze(img_tensor, 0)
img_tensor = img_tensor.permute(0, 3, 1, 2)

output = model(img_tensor)
output = output.squeeze()
output = output.permute(1,2,0)

image_array = output.detach().numpy()
plt.imshow(image_array)
plt.show()



