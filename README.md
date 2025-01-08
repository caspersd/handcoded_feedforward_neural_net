
# Overview:

This project was a hands-on exploration of implementing gradient descent to train a fully connected, feed-forward neural network from scratch. The primary objective was to deepen understanding of the math behind neural networks by directly calculating partial derivatives and applying gradient descent optimization.

The project involved creating two main neural network implementations:

- Manual Gradient Calculation: The initial network was manually coded simple feed forward network (with a fixed number of layers and neurons) where each partial derivative was calculated individually. This approach provided foundational insight into backpropagation mechanics and how gradients update weights to minimize error.

- Matrix Calculus Approach: The second implementation improved efficiency by leveraging matrix calculus for calculating partial derivatives. This enabled flexibility in defining an arbitrary number of layers and neurons per layer, overcoming the rigidity of the manual approach.

## Project Key Concepts
Gradient Descent Optimization: The network employs gradient descent to minimize the Mean Squared Error (MSE) loss function. By adjusting weights and biases through backpropagation, each layer calculates and propagates error gradients to achieve optimal model performance.

Layer Structure: The network consists of fully connected layers and an activation layer, each implementing forward, backward, and step (weight update) functions. For simplicity, the only activation function that was considered was the sigmoid activation function.

## Next Steps

A potential future enhancement could involve implementing this architecture with computational graphs using TensorFlow. While this project’s focus was to understand the mathematics underlying neural networks, a TensorFlow-based approach would facilitate scaling the model for larger datasets while taking advantage of automatic differentiation for efficiency.

## Acknowledgements

This project was developed with insights and guidance from *Math for Deep Learning* by Robert Kneusel, which served as a valuable reference for implementing mathematical concepts in the code. The book provided foundational understanding and practical examples that were integral to the project.

Much of the project took inspiration from Mr. Kneusel's work, though the code was implemented independently after an initial reading for foundational understanding. For the matrix calculus approach, minibatch training was implemented to ensure every datapoint was utilized, with each epoch representing a full iteration over the dataset.

## Reference
Kneusel, Robert. *Math for Deep Learning: What You Need to Know to Understand Neural Networks*. Manning Publications, 2021.
