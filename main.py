# imports

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as pil
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms as trans
from torchvision import models
from collections import OrderedDict
import load, imshow, extract_features, gramian
 
# setting model & inputs

device = torch.device("cuda")
content = load('Taj.jpeg').to(device)
style = load('Starry_Night.jpg',shape=content.shape[-2:]).to(device)
target = content.clone().requires_grad_(True).to(device)

vgg = models.vgg19(pretrained=True).features

for param in vgg.parameters():
    param.requires_grad_(False)

model = vgg
model.to(device)

layers = {'0': 'conv1_1',
          '5': 'conv2_1',
          '10':'conv3_1',
          '19':'conv4_1',
          '21':'conv4_2',
          '28':'conv5_1'}

for i, l in enumerate(model):
  if isinstance(l, torch.nn.MaxPool2d):
    model[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    
# extracting features from images
	
content_feat = extract_features(content, model, layers)
style_feat = extract_features(style, model, layers)
    
gram_matrix = {}
for l in style_feat:
  gram_matrix[l] = gramian(style_feat[l])

weights = {}
w = 1.
for l in layers:
  weights[layers[l]] = w
  w -= 0.15
  
content_weight = 1
style_weight = 5e3

# training loop

opt = torch.optim.Adam([target], lr=0.007)
epoch = 5000

for i in range(1, epoch+1):
    style_loss = 0
    target_feat = extract_features(target, model, layers)
    content_loss = torch.mean((target_feat['conv5_1'] - content_feat['conv5_1'])**2)
    
    for l in weights:
        target_f = target_feat[l]
        b, d, h, w = target_f.shape
        
        target_gram = gramian(target_f)
        gram_mat = gram_matrix[l]
        
        layer_style_loss = weights[l] * torch.mean((target_gram - gram_mat)**2)
        style_loss += layer_style_loss / (d * h * w)
        
    j = content_weight * content_loss + style_weight * style_loss
    
    opt.zero_grad()
    j.backward(retain_graph=True)
    opt.step()

    if i % 500 == 0:
      rounded_loss = round(j.item(), 2)
      print('Iteration:' , i , '     Total loss:' , rounded_loss)  
      imshow(target)
      plt.savefig('Starry_Taj.jpg')
