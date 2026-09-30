
# The Mathematical Intuition Behind Deep Learning
## From the Dot Product to Multivariate Calculus and the Jacobian, with Python

Alex Punnen \
&copy; All Rights Reserved 

---

## Contents

- Chapter 1: [The simplest Neural Network - Perceptron using Vectors and Dot Products](1_vectors_dot_product_and_perceptron.md)
- Chapter 2: [Perceptron Training via Feature Vectors & HyperPlane split](2_perceptron_training.md)
- Chapter 3: [Gradient Descent and Optimization](3_gradient_descent.md)
- Chapter 4: [Backpropagation with Scalar Calculus](4_backpropogation_chainrule.md)
- Chapter 5: [Backpropagation with Matrix Calculus](5_backpropogation_matrix_calculus.md)
- Chapter 6: [Backpropagation with Softmax and Cross Entropy](6_backpropogation_softmax.md)
- Chapter 7: [Neural Network Implementation](7_neuralnetworkimpementation.md)

---

## Notation

Symbols are defined in each chapter as they are introduced; a few conventions hold throughout:

- **Superscript = layer, subscript = component.** $a^2$ is the activation of layer 2; $p_i$ is the $i$-th element of vector $p$.
- **Column vectors.** A layer computes $z^l = W^l a^{l-1} + b^l$, and gradients are column vectors, so an error is pushed back through a layer as $(W^l)^T \delta^l$.
- **$C$** is the cost (loss) and **$\eta$** is the learning rate.
- Chapter 7's NumPy code uses the row-vector / batch convention (`X @ W`); the note at the start of that chapter explains the transposition.
