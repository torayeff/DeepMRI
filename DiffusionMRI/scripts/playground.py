import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)

for n, p in resnet18.named_parameters():
    print(n)
