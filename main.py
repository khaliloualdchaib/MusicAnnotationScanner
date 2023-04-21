from Dataloader import *
from toTesnor import ToTensor
from patches import *
# Iterate through tensors
"""
dataloader = FAAMDataset("coco-1678101495.6258435.json",transform=transforms.Compose([ToTensor()]))

for i in range(7,10):
    image = dataloader[i]
    print(image)
"""

# Iterate through images

dataloader = FAAMDataset("new.json")
#dataloader.transform = transform=transforms.Compose([Rescale(dataloader.getAverageSize())])
image = dataloader[9]
plt.imshow(image)
plt.savefig("image.png")
p = Patch(dataloader, 500, 500)
p.CreatePatchesImage(image,9)
patches = torch.load('Patches41Image.pt')

p.ShowImage(patches[0]["image"], True, "ShowImageTest.png")

"""
for i in range(7,10):
    image = dataloader[i]
    # Show the image using matplotlib
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
"""