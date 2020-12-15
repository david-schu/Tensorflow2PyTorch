# Tensorflow2Torch
 Example of how to convert a Tensorflow (1) model to a Pytorch model
 
## Heads up
To use this guideline, the tensorflow model architecture is needed as well as pretrained weights and biases as checkpoint (ckpt) files.
This was not tested for a tensorflow 2 model and minor changes might be necessary to make things work. 

### Prerequisites
- Python3 and following packages: Tensorflow2, PyTorch
- to run tensorflow1 code as in example, disable v2 behaviour as shown below

```
import tensorflow

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

```

During training of your tensorflow model, a checkpoint can be saved using the tensorflow saver:

```
global_step = tf.contrib.framework.get_or_create_global_step()
saver = tf.train.Saver(max_to_keep=3)
saver.save(sess,
           os.path.join(model_dir, 'checkpoint'),
           global_step=global_step)
```


### Example Description
In the following I am going to explain the steps that are needed to convert a pretrained Tensorflow model to a PyTorch model. 
These steps are implemented in the [tensorflow2torch](https://github.com/david-schu/Tensorflow2Torch/blob/master/tensorflow2pytorch.py) file of this repo.

#### 1. Create a PyTorch model with the same architecture as the tensorflow model
To transfer the weights of a tensorflow model, its architecture needs to be replicated by a PyTorch model. Note that PyTorch uses different tensor structures than Tensorflow. Also Pytorch uses different padding in convolutional layers. Refer to the tensorflow and PyTorch docu to match parameters. Depending on the models architecture this can be time intensive. To see an example of equivalent models refer to the [Tensorflow model](https://github.com/david-schu/Tensorflow2Torch/blob/master/tfModels/tfModel.py)
and [PyTorch model](https://github.com/david-schu/Tensorflow2Torch/blob/master/torchModel.py) of this repo.

#### 2. Load the Tensorflow model and the latest checkpoint to initialize a pretrained model
```
sess = tf.Session()
tf_model = tfModel.Model()
checkpoint = tf.train.latest_checkpoint('./tfModels/adv_trained')
restorer = tf.train.Saver()
restorer.restore(sess, checkpoint)
```

#### 3. Get the trainable variables
Print the trainable variables of each layer and their shapes. This is a good point to check if the the number of trainable parameters in both models are equal.
Store the variable names to use them for weight and bias extraction later.
```
for val in tf.trainable_variables():
    print(val.name + ': shape=' + str(val.shape))
```

#### 4. Assign weights and biases to new model
Use the previously fetched variable names to extract the parameters from the tensorflow model via the session.run() function. Convert the parameters to torch tensors and permute if necessary to match the PyTorch models shape. Then convert the weights tensor to a torch parameter tensor which has a gradient and is trainable.
```
weights_cv1 = torch.from_numpy(sess.run('Variable:0')).permute((3, 2, 0, 1))
model_madry.conv1.weight = torch.nn.Parameter(weights_cv1)
```

#### 5. Test or retrain your model
Now you can run a test to see if accuracies match. 

#### 6. Congratulations you're Done!

## Authors

* **David Schultheiß** - *Initial work* - [T2T](https://github.com/david-schu/Tensorflow2Torch/)

## Acknowledgments

* Dylan Paiton, Bethge Lab
* Medium article by Thomas Wolf [ From TensorFlow to PyTorch](https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28)
* etc
