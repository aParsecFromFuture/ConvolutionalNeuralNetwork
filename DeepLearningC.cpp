#include <iostream>
#include "cnn.h"

int main()
{
    int i, j, k, l;
    int epoch, sample_count, train_sample_count, validation_sample_count, test_sample_count, batch_size, batch_count, validation_batch_count;
    int* shuffle_array;
    int* train_label, * test_label;
    float err, validation_split;
    float* out0, * out1, * out2, * out3, * out4, * out5, * out6, * out7, * out8, * tar0;
    float* mean, * var, * train_data, * test_data, * train_x, * train_y, * valid_x, * valid_y, * test_x, * test_y;
    
    const char* train_x_path = "./train_x.txt";
    const char* train_y_path = "./train_y.txt";
    const char* test_x_path = "./test_x.txt";
    const char* test_y_path = "./test_y.txt";

    srand(time(0));

    epoch = 10;
    sample_count = 10000;
    test_sample_count = 100;
    validation_split = 0.2f;
    batch_size = 32;

    LEARNING_RATE = 0.1f;
    MOMENTUM = 0.9f;
    BATCH_DIV = 1.0f / batch_size;  // For reducing the computational cost of division operation

    train_sample_count = (1.0f - validation_split) * sample_count;
    validation_sample_count = validation_split * sample_count;
    batch_count = train_sample_count / batch_size;
    validation_batch_count = validation_sample_count / batch_size;
    shuffle_array = create_shuffle_index(sample_count);

    mean = (float*)malloc(sizeof(float) * SHAPE3(1, 28, 28));
    var = (float*)malloc(sizeof(float) * SHAPE3(1, 28, 28));

    // CNN MODEL ARCHITECTURE

    struct conv         layer1 = create_conv(3, 3, 32, 28, 28, 1, batch_size);
    struct batchnorm    layer2 = create_batchnorm(26, 26, 32, batch_size);
    struct maxp         layer3 = create_maxp(26, 26, 32, batch_size);
    struct dense        layer4 = create_dense(13 * 13 * 32, 128, batch_size);
    struct batchnorm    layer5 = create_batchnorm(128, 1, 1, batch_size);
    struct dense        layer6 = create_dense(128, 10, batch_size);
    
    tar0 = init_mem(SHAPE2(10, batch_size));
    out0 = init_mem(SHAPE4(28, 28, 1, batch_size));
    out1 = init_mem(SHAPE4(26, 26, 32, batch_size));
    out2 = init_mem(SHAPE4(26, 26, 32, batch_size));
    out3 = init_mem(SHAPE4(26, 26, 32, batch_size));
    out4 = init_mem(SHAPE4(13, 13, 32, batch_size));
    out5 = init_mem(SHAPE2(128, batch_size));
    out6 = init_mem(SHAPE2(128, batch_size));
    out7 = init_mem(SHAPE2(128, batch_size));
    out8 = init_mem(SHAPE2(10, batch_size));

    // TRAINING STAGE

    printf("1. Training data is loading...\n");

    train_data = read_data(train_x_path);
    train_label = read_idata(train_y_path);

    train_x = batch_normalization_train(train_data, train_sample_count, SHAPE3(28, 28, 1), mean, var);
    valid_x = batch_normalization_test(&train_data[train_sample_count * SHAPE3(28, 28, 1)], validation_sample_count, SHAPE3(28, 28, 1), mean, var);
    free(train_data);

    train_y = create_target(10, train_sample_count, train_label);
    valid_y = create_target(10, validation_sample_count, &train_label[train_sample_count]);
    free(train_label);

    // TRAINING STAGE

    printf("2. Training stage has started\n\n");
    
    for (i = 0; i < epoch; i++) {
        err = 0.0f;
        for (j = 0; j < batch_count; j++) {
            copy_mem(train_x, out0, j * batch_size, batch_size, SHAPE3(28, 28, 1), shuffle_array);
            copy_mem(train_y, tar0, j * batch_size, batch_size, SHAPE1(10), shuffle_array);
            
            f_conv(out0, &layer1, out1);
            f_batchnorm(out1, &layer2, out2);
            f_relu(out2, SHAPE4(26, 26, 32, batch_size), out3);
            f_maxp(out3, &layer3, out4);
            f_dense(out4, &layer4, out5);
            f_batchnorm(out5, &layer5, out6);
            f_relu(out6, SHAPE2(128, batch_size), out7);
            f_output(out7, &layer6, out8);
            
            err += evaluate(out8, tar0, 10, batch_size);

            b_output(out8, tar0, &layer6, out7);
            b_relu(out7, SHAPE2(128, batch_size), out6);
            b_batchnorm(out6, &layer5, out5);
            b_dense(out5, &layer4, out4);
            b_maxp(out4, &layer3, out3);
            b_relu(out3, SHAPE4(26, 26, 32, batch_size), out2);
            b_batchnorm(out2, &layer2, out1);
            b_conv(out1, &layer1, out0);
        }
        printf("epoch %d: ", i + 1);
        printf("error: %.4f\t", err / batch_count);
        shuffle(shuffle_array, train_sample_count);

        // VALIDATION STAGE

        err = 0.0f;

        for (j = 0; j < validation_batch_count; j++) {
            copy_mem(valid_x, out0, j * batch_size, batch_size, SHAPE3(28, 28, 1), 0);
            copy_mem(valid_y, tar0, j * batch_size, batch_size, SHAPE1(10), 0);

            f_conv(out0, &layer1, out1);
            f_batchnorm(out1, &layer2, out2);
            f_relu(out2, SHAPE4(26, 26, 32, batch_size), out3);
            f_maxp(out3, &layer3, out4);
            f_dense(out4, &layer4, out5);
            f_batchnorm(out5, &layer5, out6);
            f_relu(out6, SHAPE2(128, batch_size), out7);
            f_output(out7, &layer6, out8);

            err += evaluate(out8, tar0, 10, batch_size);
        }
        printf("validation_error: %.4f\n", err / validation_batch_count);
    }

    free(train_x);
    free(train_y);
    free(valid_x);
    free(valid_y);
    free(shuffle_array);

    // TESTING STAGE

    printf("\n3. Test data is loading...\n");
    
    test_data = read_data(test_x_path);
    test_label = read_idata(test_y_path);
    
    test_x = batch_normalization_test(test_data, test_sample_count, SHAPE3(28, 28, 1), mean, var);
    free(test_data);
    
    test_y = create_target(10, test_sample_count, test_label);
    free(test_label);

    batch_count = test_sample_count / batch_size;

    printf("4. Test stage has started\n\n");

    for (i = 0; i < batch_count; i++) {
        copy_mem(test_x, out0, i * batch_size, batch_size, SHAPE3(28, 28, 1), 0);
        copy_mem(test_y, tar0, i * batch_size, batch_size, SHAPE1(10), 0);

        f_conv(out0, &layer1, out1);
        f_batchnorm(out1, &layer2, out2);
        f_relu(out2, SHAPE4(26, 26, 32, batch_size), out3);
        f_maxp(out3, &layer3, out4);
        f_dense(out4, &layer4, out5);
        f_batchnorm(out5, &layer5, out6);
        f_relu(out6, SHAPE2(128, batch_size), out7);
        f_output(out7, &layer6, out8);

        for (j = 0; j < batch_size; j++) {
            printf("Sample %d\n", i * batch_size + (j + 1));
            for (k = 0; k < 10; k++)
                printf("%.2f - %.2f\n", tar0[j * 10 + k], out8[j * 10 + k]);
        }
    }
}
