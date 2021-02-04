# ConvolutionalNeuralNetwork
Convolutional neural network from scratch with handwritten digit recognition example

## Features

* Supported methods
  * Mini-batch gradient
  * Batch gradient descent
* Supported layers
  * Convolutional layer
  * Max pooling layer
  * Batch normalization layer
  * Dense layer
* Supported optimizers
  * Momentum
* Supported preprocessing operations
  * Data shuffling
  * Z-score normalization
* Supported loss functions
  * Categorical cross entropy
* Multi-layer convolution support
* Multi channel input support
* Validation split support

## Convolutional layer

### Feed Forward

![alt text](github%20resource/f_conv.png)


### Updating weights

![alt text](github%20resource/train_conv.png)


### Backpropagation

![alt text](github%20resource/b_conv.png)

## Max Pooling Layer

### Feed Forward

![alt text](github%20resource/f_maxpool.png)


### Backpropagation

![alt text](github%20resource/b_maxpool.png)

## Hidden Dense Layer

### Feed Forward

![alt text](github%20resource/f_dense.png)


### Updating weights

![alt text](github%20resource/train_dense.png)


### Backpropagation

![alt text](github%20resource/b_dense.png)

## Output Dense Layer

### Feed Forward

![alt text](github%20resource/f_output.png)


### Updating weights

![alt text](github%20resource/train_output.png)


### Backpropagation

![alt text](github%20resource/b_output.png)

### References

* Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift. Google, 1600 Amphitheatre Pkwy, Mountain View, CA 94043.

* Frederik Kratzert (Feb 12, 2016). Understanding the backward pass through Batch Normalization Layer. 
https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html.

* Pierre Jaumier (Jul 10, 2019). Backpropagation in a convolutional layer.
https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509

* Emily Elia (Jul 29, 2019). A Guide to Building Convolutional Neural Networks from Scratch.
https://towardsdatascience.com/a-guide-to-convolutional-neural-networks-from-scratch-f1e3bfc3e2de

* Mayank Agarwal (Dec 14, 2017). Back Propagation in Convolutional Neural Networks — Intuition and Code.
https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199

* Jason Brownlee (Jan 23, 2019). How to Configure the Learning Rate When Training Deep Learning Neural Networks.
https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/
