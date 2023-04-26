from FAAMDataset import *
from toTesnor import ToTensor
from patches import *
from PatchDataset import *

#Put here the path to root folder
root = 'C:\\Users\\Redux Gamer\\Documents\\MusicAnnotationScanner'
#dataset = FAAMDataset("new.json")
#dataloader.transform = transform=transforms.Compose([Rescale(dataloader.getAverageSize())])
#p = Patch(dataset, 500, 500, root)
#p.CreatePatches()
#print(patches)
#p.ShowImage(patches["image"], True, "ShowImageTest.png")

data = PatchDataset("Patches.csv")
data.ShowImage(data[0][0])