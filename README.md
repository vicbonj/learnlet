# learnlet

This code is the PyTorch implementation of the Learnlet transform originally developed in Ramzi et al., 2020 (https://ieeexplore.ieee.org/document/9287317) and modified in Bonjean et al., 2025 (arxiv link to come).

The learnlets have been trained on 10,000 images from the ImageNet dataset, and the weights for the default value parameters of the network are automatically loaded when the class is instantiated. Hence the use is very straightforward:

`from learnlet import Learnlet`
`learnlet = Learnlet()`


## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

Author: Victor Bonjean

Mail: victor.bonjean40@gmail.com
