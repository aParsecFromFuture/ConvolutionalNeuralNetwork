#include <iostream>
#include "cnn.h"

int main() {
    srand(time(0));
    
    printf("Training data is loading...\n");
    
    ImageArray train_images(28, 28, 1);
    train_images.load_from("mnist_train_x.txt");
    train_images.split(0.1);
    
    LabelArray train_labels(10);
    train_labels.load_from("mnist_train_y.txt");
    
    ImageArray test_images(28, 28, 1);
    test_images.load_from("mnist_test_x.txt");

    LabelArray test_labels(10);
    test_labels.load_from("mnist_test_y.txt");
    
    ImageArray::min_max_scaling(train_images);
    ImageArray::min_max_scaling(test_images);
    
    printf("Training stage has started\n\n");
    
    CNN model;
    
    model.add_layer(new Conv2D(5, 5, 16));
    model.add_layer(new Relu());
    model.add_layer(new MaxPool());
    model.add_layer(new BatchNorm());
    model.add_layer(new Conv2D(5, 5, 32));
    model.add_layer(new Relu());
    model.add_layer(new MaxPool());
    model.add_layer(new BatchNorm());
    model.add_layer(new Dense(128));
    model.add_layer(new Relu());
    model.add_layer(new BatchNorm());
    model.add_layer(new Output(10));
    
    model.setup(28, 28, 1, 64, 10);
    
    model.train(train_images, train_labels, 1, 0.1f, 0.9f);
    LabelArray prediction = model.test(test_images);

    model.save_to("my_model.txt");
    
    /*
    CNN model;
    model.load_from("my_model.txt");
    model.test(test_images, test_labels);
    */
}
