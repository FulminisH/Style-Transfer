def gramian(tensor):

    b,d,h,w = tensor.shape
    tensor = tensor.view(d,h*w)
    gram = torch.mm(tensor, tensor.t())
    
    return gram 