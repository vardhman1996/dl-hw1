from uwnet import *

def softmax_model():
    l = [make_connected_layer(3072, 10, SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(3072, 32, LRELU),
            make_connected_layer(32, 10, SOFTMAX)]
    return make_net(l)

def conv_net():
    # How many operations are needed for a forard pass through this network?
    # Your answer: 1,108,480
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10, SOFTMAX)]
    return make_net(l)

def your_net():
    # Define your network architecture here. It should have 5 layers. How many operations does it need for a forward pass?
    # It doesn't have to be exactly the same as conv_net but it should be close.

    # Has the same number of operations as CONVNET
    l = [   make_connected_layer(3072, 72, LRELU),
            make_connected_layer(72, 512, LRELU),
            make_connected_layer(512, 1104, LRELU),
            make_connected_layer(1104, 256, LRELU),
            make_connected_layer(256, 10, SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test", "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
# Convnet test accuracy: 65%, Fully connected network test accuracy: 51%
# The fully connected netowork uses all the pixels and channels of an image and tries to find a relationship. However, only
# the neighboring pixels give the most useful information about a particular pixel and convolutions are best suited for this. 
# When using the same number of operations, the convnet uses these operations most efficiently using convolutions 
# unlike the fully connected network and hence results in a higher accuracy from convnet on this dataset. The fully connected 
# network might need many more operations to get similar accuracy.

