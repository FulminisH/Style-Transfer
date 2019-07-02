def extract_features(image, model, layers):

    features = {}
    x = image.to(device)
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features