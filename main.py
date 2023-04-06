from Dataloader import *
from toTesnor import ToTensor
from Rescale import *
from torchvision import transforms
from patches import *
# Iterate through tensors
"""
dataloader = FAAMDataset("coco-1678101495.6258435.json",transform=transforms.Compose([ToTensor()]))

for i in range(7,10):
    image = dataloader[i]
    print(image)
"""

# Iterate through images

dataloader = FAAMDataset("coco-1678101495.6258435.json")
dataloader.transform = transform=transforms.Compose([Rescale(dataloader.getAverageSize())])
image = dataloader[4]
p = Patch(dataloader, 500, 500)
p.CreatePatchesImage(image)
"""
for i in range(7,10):
    image = dataloader[i]
    # Show the image using matplotlib
    plt.imshow(image.permute(1, 2, 0))
    plt.show()"""