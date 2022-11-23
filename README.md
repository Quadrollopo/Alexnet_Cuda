# Alexnet in Cuda

While the goal was to implement alexnet in cuda, you can create all the architecture you want, using convolutional, maxpool and fully connected layer.  
[Here](https://github.com/Quadrollopo/Alexnet_Cuda/blob/e795106c6237c11ccc3f251b51c738eca2429cd6/main.cu#L27) there is an exemple of a network with 2 conv layers and 3 FC layers.

## Creating a network
You can create a network by calling:  
```Network net(num_neurons, lr)``` if you want create a network with only FC  
or  
```Network net(img_side, channels, lr)``` if you want create a network with the first layer convolutional  
Where  
- _num_neurons_ is the number of neurons of the first layer
- _img_side_ is the side of the input image, so if you want to import an image 24x24 put 24 here. You can only work on squared images
- _channels_ the channels of the input image, in a nutshell, 1 if the image is black and white and 3 if is colored
- _lr_ the learning rate  
  
You can then add new layer by calling the proper function to the network  
- `net.addConvLayer(int kern_size, int num_kernels, int stride, bool pad, Act func)` if pad  is true will pad the image with kern_size-1, and func is the activation function choose between reLu, sigmoid and softmax
- `net.addFullLayer(int neurons, Act func)` where _neurons_ is the number of neuron of the layer
- `net.addPoolLayer(int pool_size, int stride)` where _pool_size_ is the size of the max pooling kernel, also this is a square

