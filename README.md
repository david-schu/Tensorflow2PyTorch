# Tensorflow2Torch
 Example of how to convert a Tensorflow (1) model to a Pytorch model

### Prerequisites
- Python3 and following packages: Tensorflow2, PyTorch
- to run tensorflow1 code as in example, disable v2 behaviour as shown below

```
import tensorflow

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

```


### Example Description
In the following I am going to explain the steps that are needed to convert a pretrained Tensorflow model to a PyTorch model.



## Authors

* **David Schultheiß** - *Initial work* - [T2T](https://github.com/david-schu/Tensorflow2Torch/)


## Acknowledgments

* Dylan Paiton, Bethge Lab
* Medium article by Thomas Wolf [ From TensorFlow to PyTorch](https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28)
* etc
