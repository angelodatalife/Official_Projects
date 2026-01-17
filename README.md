![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project | Deep Learning: Image Classification using CNN and Transfer Learning

## Task Description

In this project, students will first build a **Convolutional Neural Network (CNN)** model from scratch to classify images from a given dataset into predefined categories. Then, they will implement a **transfer learning approach** using a pre-trained model. Finally, students will **compare the performance** of the custom CNN and the transfer learning model based on evaluation metrics and analysis.

## Dataset

The dataset for this task is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. You can download the dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html).


## Assessment Components

1.  **Data Preprocessing**
    
    *   Data loading and preprocessing (e.g., normalization, resizing, augmentation).
    *   Create visualizations of some images, and labels.
2.  **Model Architecture**
    
    *   Design a CNN architecture suitable for image classification.
    *   Include convolutional layers, pooling layers, and fully connected layers.
3.  **Model Training**
    
    *   Train the CNN model using appropriate optimization techniques (e.g., stochastic gradient descent, Adam).
    *   Utilize techniques such as early stopping to prevent overfitting.
4.  **Model Evaluation**
    
    *   Evaluate the trained model on a separate validation set.
    *   Compute and report metrics such as accuracy, precision, recall, and F1-score.
    *   Visualize the confusion matrix to understand model performance across different classes.
5.  **Transfer Learning**
    
    *   Perform transfer learning with your chosen pre-trained models i.e., you will probably try a few and choose the best one (e.g., VGG16, Inception, ResNet trained on ImageNet)  
        *   You may find this [link](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub) helpful.
        *   [This](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) is the Pytorch version.
    *   Train and evaluate the transfer learning model.
    *   Compare its performance against your custom CNN.
    *   Discuss advantages and trade-offs of using transfer learning over building a model from scratch.
        
6.  **Code Quality**
    
    *   Well-structured and commented code.

## Submission Details

*   Deadline for submission: end of the week or as communicated by your teaching team.
*   Submit the following:
    1.  Python code files (`*.py`, `ipynb`) containing the model implementation and training process.
    2.  Any additional files necessary for reproducing the results (e.g., requirements.txt, README.md).
    3.  PPT presentation

## Additional Notes

*   Students are encourage to experiment with different architectures, hyper-parameters, and optimization techniques.
*   Provide guidance and resources for troubleshooting common issues during model training and evaluation.
*   Students will discuss their approaches and findings in class during assessment evaluation sessions.