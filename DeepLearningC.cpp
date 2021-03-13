#include <iostream>
#include "ann.h"

int main() {
    srand(time(0));
    
    printf("Training data is loading...\n");
    
    ImageArray train_images(28, 28, 1);
    train_images.load_from("mnist_train_x.txt");
    
    LabelArray train_labels(10);
    train_labels.load_from("mnist_train_y.txt");
    
    ImageArray test_images(28, 28, 1);
    test_images.load_from("mnist_test_x.txt");

    LabelArray test_labels(10);
    test_labels.load_from("mnist_test_y.txt");
    
    ImageArray::min_max_scaling(train_images);
    ImageArray::min_max_scaling(test_images);
    
    printf("Training stage has started\n\n");
    
    ANN model;
    
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
    model.add_layer(new Dense(10));
    model.add_layer(new Softmax());

    model.setup(28, 28, 1, 64, 10);
    
    model.train(train_images, train_labels, 1, 0.1f, 0.1f, 0.9f);
    
    int* numbers = test_labels.simplify();
    int* prediction = model.test(test_images).simplify();

    for (int i = 0; i < test_labels.length(); i++)
        printf("num : %d \t pred : %d\n", numbers[i], prediction[i]);
    
    model.save_to("my_model.txt");
    
    /*
    CNN model;
    model.load_from("my_model.txt");
    model.test(test_images);
    */
}
