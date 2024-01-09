from extractor.ctran import *
import torch.nn as nn
import torch
from ultralytics import YOLO

model = ctranspath()
model.head = nn.Identity()
td = torch.load(r'C:/Users/minki.kim/Desktop/org/extractor/ctranspath.pth')
model.load_state_dict(td['model'], strict=True)
modules = list(model.children())[:-2]
ctran_extractor = torch.nn.Sequential(*modules)
print(ctran_extractor)
