# Style-Transfer
Contains an implementation of neural style transfer in PyTorch. Style Transfer was first implemented by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge in their paper 'A Neural Algorithm of Artistic Style'. https://arxiv.org/abs/1508.06576

The model uses VGGNet pretrained model layers with replacing Max pooling with Average pooling to extract the features from the content image and style image. Then, it applies the artistic intricacies of the style image on the content image to produce a target image the mixes between both style and content.
