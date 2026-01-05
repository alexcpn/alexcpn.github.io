


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

A common choice is the **Mean Squared Error (MSE)**:

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

## Take 2 - Deeper Dive


The Error function is a scalar function of the weights and biases. 

 Whereas for a simple linear regression, the same Mean Square Error function, the paramters are just one layer feature vectors, the function is a convex funciton (see left in picture)

Whereas for a deep learning model the function is a composition of multiple non-linear functions, the cost function is not a convex function but rather consists of multiple saddle points and local minima. (see right in picture)

![costfunction]

**How will Gradient Descent work in this case - non convex function?**

Gradient descent does not attempt to find the global minimum, but rather follows the local slope of the cost function and converges to a local minimum or a flat region.

The cost function is a scalar function of the weights and biases and is differentiable almost everywhere*. At any point in parameter space, the gradient indicates the direction of steepest local increase, and moving in the opposite direction reduces the cost. During optimization, the algorithm may encounter local minima or saddle points.

(*The function is not differentiable at the point where the function is zero ex ReLU. This is not a problem in practice, as optimization algorithms handle such points using [subgradients](images/subgradient.png))

In practice, deep learning works well despite non-convexity, partly because modern networks have millions of parameters and their loss landscapes contain many saddle points and wide, flat minima rather than [poor isolated local minima](images/poorlocalminima.png).

Furthermore, we rarely use full-batch gradient descent. Instead, we use variants such as Stochastic Gradient Descent (SGD) or mini-batch gradient descent.

In these methods, gradients are computed using a single training example or a small batch of examples rather than the entire dataset. The resulting gradient is an average over the batch and serves as a noisy approximation of the true gradient. This stochasticity helps the optimizer escape saddle points and sharp minima, enabling effective training in practice.

**What does it mean to take the derivative of a scalar function with respect to vector-valued parameters?**

We are trying to find the minimum of this function with respect to the weights $w$ and biases $b$ of the last layer here. 

As the last layer inherits from all the previous layers, the minimum of the cost function with respect to the weights and biases of the whole network is what we calcuate with gradient descent. This is a not a number/scalar but a gradient vector which is a coordinate  representation of the underlying linear map

This is important. How we then update the weights in all the previous layers propogating the gradients backward is the topic of backpropagation which we will see later. This is Automatic Differentiation in reverse mode using the chain rule and making use of the model of the neural networ as a composition of functions.


Lets focus still on the gradient descent for now.

The weight is not a scalar but a vector. (represented as matrix)

This means a level of complexity which in usual explantion is not shown.

We have seen that Gradinet descent uses first order derivatives to find the minimum of the function.

What is a derivative of a vector function? This is something hard to explain in a simple way. It is a matrix of gradients.

This paper - The Simple Essence of Automatic Differentiation
Extended version Conal Elliott explains this in detail.
Quoting from it below.

---

The derivative $f'(x)$ of a function $f: \mathbb{R} \to \mathbb{R}$ at a point $x$ (in the domain of $f$) is a number, defined as follows:

$$
f'(x) = \lim_{\epsilon \to 0} \frac{f(x + \epsilon) - f(x)}{\epsilon}
$$

That is, $f'(x)$ tells us how fast $f$ is scaling input changes at $x$.

Note here that $\mathbb{R}$ is the set of real numbers and $\epsilon$ is also a real number.

How well does this definition hold up beyond functions of type $\mathbb{R} \to \mathbb{R}$? 

When we extend to $\mathbb{R}^m \to \mathbb{R}^n$, this definition no longer makes sense, as it would rely on dividing by a vector $\epsilon \in \mathbb{R}^m$.

This difficulty of differentiation with non-scalar domains is usually addressed with the notion of "partial derivatives" with respect to the $m$ scalar components of the domain $\mathbb{R}^m$.

When the codomain $\mathbb{R}^n$ is also non-scalar (i.e., $n > 1$), we have a matrix $J$ (the Jacobian), with $J_{ij} = \partial f_i / \partial x_j$ for $i \in \{1, \dots, n\}$, where each $f_i$ projects out the $i$-th scalar value from the result of $f$.

Moreover, each of these situations has an accompanying chain rule, which says how to differentiate the composition of two functions. Where the scalar chain rule involves multiplying two scalar derivatives, the vector chain rule involves "multiplying" two matrices $A$ and $B$ (the Jacobians), defined as follows:

$$
(A \cdot B)_{ij} = \sum_{k=1}^m A_{ik} \cdot B_{kj}
$$

Since one can think of scalars as a special case of vectors, and scalar multiplication as a special case of matrix multiplication, perhaps we've reached the needed generality.

The derivative of a function $f: a \to b$ at some value in $a$ is thus not a number, vector, matrix, or higher-dimensional variant, but rather a linear map (also called "linear transformation") from $a$ to $b$, which we will write as $a \multimap b$. 

The numbers, vectors, matrices, etc. mentioned above are all different representations of linear maps; and the various forms of "multiplication" appearing in their associated chain rules are all implementations of linear map composition for those representations.

---

This above passage is taken from the paper on auto differentiation by Conal Elliott. Why I put it here is that it applies to our case of gradient descent as well. 

So the Loss function is a funciton of weight vectors $w$ and bias vectors $b$. 

$$
C(w, b)
$$

Minimizing this function with respect to $w$ and $b$ is the goal of training. That is mathematically we try to find the slope of the function from an arbitary selected point in the weight space and incrementally "go down" the gradient.

For this we need to use the mathematical tool of derivatives to calculate "the gradeint" of the cost funciton at the selected arbitary weight vector $w$ and bias vector $b$.

This gradient is a vector of partial derivatives of the cost function with respect to each weight and bias.

$$
\nabla C = \begin{bmatrix}
\frac{\partial C}{\partial w} \\
\frac{\partial C}{\partial b}
\end{bmatrix}
$$

And it means a linear transformation from the weight space to the cost space.


## What's Next?

The big question remains: **How do we calculate these gradients $\frac{\partial C}{\partial w}$ efficiently for all the layers?**

This is where the **Backpropagation** algorithm comes in.

[neuralnetwork]: https://i.imgur.com/gE3QKCf.png
[costfunction]: images/costfunction.png