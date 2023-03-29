from Dataloader import *
from toTesnor import ToTensor


# Iterate through tensors
"""
dataloader = FAAMDataset("coco-1678101495.6258435.json",transform=transforms.Compose([ToTensor()]))

for i in range(7,10):
    image = dataloader[i]
    print(image)
"""

# Iterate through images
"""
dataloader = FAAMDataset("coco-1678101495.6258435.json",)

for i in range(7,10):
    image = dataloader[i]

    # Show the image using matplotlib
    plt.imshow(image)
    plt.show()
"""
