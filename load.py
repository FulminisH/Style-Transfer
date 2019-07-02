def load(path, shape=None):
  ''' Input: image path
      Output: transformed - resized and normalized - image '''
  image = pil.open(path).convert('RGB')
  max_size = 1080
  if max(image.size) >= max_size:
      size = max_size
  else:
      size = max(image.size)
      
  if shape is not None:
        size = shape
  
  transform = trans.Compose([trans.Resize(size),trans.ToTensor(),
                            trans.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

  image = transform(image)[:3,:,:].unsqueeze(0)
  return image