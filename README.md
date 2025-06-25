# learnlet

This code is the PyTorch implementation of the Learnlet transform originally developed in [Ramzi et al., 2020](https://link-to-author-profile-or-paper) and modified in Bonjean et al., 2025 (arxiv link to come soon).

The learnlets have been trained on 10,000 images from the ImageNet dataset, and the weights for the default value parameters of the network are automatically loaded when the class is instantiated. Hence the use is very straightforward and here is a small example:

<pre>python\n
  from learnlet import Learnlet
  import torch

  print('coucou')
\n
</pre>
`from learnlet import Learnlet`<br/>
`import torch`<br/>
`from skimage import data, transform, img_as_float32`<br/><br/>
`#Import example image Y`<br/>
`img = data.camera()`<br/>
`img_256 = transform.resize(img, (256, 256), anti_aliasing=True)`<br/>
`Y = torch.from_numpy(img_as_float32(img_256))[None,None,:]`<br/><br/>
`#Add noise`<br/>
`sigma = torch.rand(1)*0.1 #Noise value`<br/>
`noise = torch.randn(Y.shape) * sigma`<br/>
`Y_noisy = Y + noise`<br/>
<br/><br/>
`#Apply transform to denoise`<br/>
`learnlet = Learnlet()`<br/>
`Y_denoised = learnlet(Y_noisy, sigma)`


## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

Author: Victor Bonjean

Mail: victor.bonjean40@gmail.com
