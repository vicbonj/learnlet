# learnlet

This code is the PyTorch implementation of the Learnlet transform originally developed in [Ramzi et al., 2020](https://link-to-author-profile-or-paper) and modified in Bonjean et al., 2025 (arxiv link to come soon).

The learnlets have been trained on 10,000 images from the ImageNet dataset, and the weights for the default value parameters of the network are automatically loaded when the class is instantiated. Hence the use is very straightforward and here is a small example:

```python
from learnlet import Learnlet
import torch
from skimage import data, transform, img_as_float32

#Import an example image Y
img = data.camera()
img_256 = transform.resize(img, (256, 256), anti_aliasing=True)
Y = torch.from_numpy(img_as_float32(img_256))[None,None,:]

#Add noise
sigma = torch.rand(1)*0.1 #Noise value
noise = torch.randn(Y.shape) * sigma
Y_noisy = Y + noise

#Apply the Learnlet transform to denoise
learnlet = Learnlet()
Y_denoised = learnlet(Y_noisy, sigma)
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

Author: Victor Bonjean

Mail: victor.bonjean40@gmail.com
