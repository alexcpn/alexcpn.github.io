


## Neural Network as a Chain of Functions

To understand deep learning, we need to understand the concept of a neural network as a chain of functions. 

A Neural Network is essentially a chain of functions. It consists of a set of inputs connected through 'weights' to a set of activation functions, whose output becomes the input for the next layer, and so on.

![neuralnetwork]

### The Forward Pass

Let's consider a simple two-layer neural network.

*   $x$: Input vector
*   $y$: Output vector (prediction)
*   $L$: Number of layers
*   $w^l, b^l$: Weights and biases for layer $l$
*   $a^l$: Activation of layer $l$ (we use sigmoid $\sigma$ here)

The flow of data (Forward Pass) can be represented as:

$$
 x \rightarrow a^{1} \rightarrow \dots \rightarrow a^{L} \rightarrow y
$$

For any layer $l$, the activation $a^l$ is calculated as:

$$
  a^{l} = \sigma(w^l a^{l-1} + b^l)
$$

where $a^0 = x$ (the input).

The linear transformation:

$z = w^T x + b$ 

defines a hyperplane (decision boundary) in the feature space.

The activation function then introduces *non-linearity*, allowing the network to combine **multiple such hyperplanes into complex decision boundaries.**

####  Why Non-Linearity Is Non-Negotiable

Without activation:

$$
f(x) = W_L W_{L-1} \dots W_1 x
$$

This collapses to:

$$
f(x) = Wx
$$

Still one big linear transformation and hence one hyperplane; the problems of not able to seperate features will come. Only because of non-linearity, we can get  multiple hyperplanes and hence a composeable complex decision boundaries that can seperate features.


So the concept of Vectors, Matrices and Hyperplanes remain the same as before. Let us explore the chain of functions part here

A neural network with $L$ layers can be represented as a nested function:$$f(x) = f_L(...f_2(f_1(x))...)$$

Each "link" in the chain is a layer performing a linear transformation followed by a non-linear activation and cascading to the final output.


### The Cost Function (Loss Function)

To train this network, we need to measure how "wrong" its predictions are compared to the true values. We do this using a **Cost Function** (or Loss Function).

A simplest Loss function is just the difference between the predicted output and the true output ($ y(x) - a^L(x) $.)

But usually we use the square of the difference to make it a non-negative function.

A common choice for regression is the **Mean Squared Error (MSE)**:

$$
 C = \frac{1}{2n} \sum_{x} \|y(x) - a^L(x)\|^2
$$

*   $n$: Number of training examples
*   $y(x)$: The true expected output (label) for input $x$
*   $a^L(x)$: The network's predicted output for input $x$

### The Goal of Training

The goal of training is to find the set of weights $w$ and biases $b$ that minimize this cost $C$.

This means that we need to optimise each component of the function $f(x)$ to reduce the cost propotional to its contribution to the final output. The method to do this is called **Backpropagation**. 

Before that lets see the how we calculate the amount to reduce via another method called **Gradient Descent**.


## Optimization: Gradient Descent

Now that we have a Cost Function $C(w, b)$, we need to find the minimum of this function with respect to the weights $w$ and biases $b$.

We calculate the gradient of the cost function with respect to each weight and bias:

$$
\nabla C = \begin{bmatrix}
\frac{\partial C}{\partial w} \\
\frac{\partial C}{\partial b}
\end{bmatrix}
$$

And then update the weights in the opposite direction of the gradient:

$$
w_{new} = w_{old} - \eta \frac{\partial C}{\partial w}
$$
$$
b_{new} = b_{old} - \eta \frac{\partial C}{\partial b}
$$

where $\eta$ is the **learning rate**.

### Why not Newton's Method?

You might ask, why not use faster optimization methods like **Newton's Method** (Newton-Raphson)?

Newton's method uses the second derivative (curvature) to find the minimum faster.
$$ w_{new} = w_{old} - \frac{C'(w)}{C''(w)} $$

However, for a neural network with millions of weights, calculating the second derivative (the Hessian matrix) is computationally infeasible. Gradient Descent, which uses only the first derivative, is much more scalable and efficient for Deep Learning.


## What's Next?

The big question remains: **How do we calculate these gradients $\frac{\partial C}{\partial w}$ efficiently for all the layers?**

This is where the **Backpropagation** algorithm comes in.

[neuralnetwork]: https://i.imgur.com/gE3QKCf.png