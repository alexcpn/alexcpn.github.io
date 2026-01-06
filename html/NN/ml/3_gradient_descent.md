


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

Still one big linear transformation and hence one hyperplane; the problems of not able to separate features will come. Only because of non-linearity, we can get  multiple hyperplanes and hence a composable complex decision boundaries that can separate features.


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

This means that we need to optimise each component of the function $f(x)$ to reduce the cost proportional to its contribution to the final output. The method to do this is called **Backpropagation**. It helps us calculate the **gradient** of the cost function with respect to each weight and bias.

Once the gradient is calculated, we can use **Gradient Descent** to update the weights in the opposite direction of the gradient.

Gradient descent is a simple optimization algorithm that works by iteratively updating the weights in the opposite direction of the gradient.

However neural network is a composition of vector spaces and linear transformations.  Hence gradient descent acts on a very complex space.

There are two or three facts to understand about gradient descent:

1. It does not attempt to find the **global minimum**, but rather follows the **local slope** of the cost function and converges to a local minimum or a flat region. **Saddle point** is a good optimisation point.

2. Gradients can **vanish or explode**, leading to slow or unstable convergence. The practical solution to control this is to use **learning rate** and using **adaptive learning rate** methods like **Adam** or **RMSprop**.

3. **Batch Size matters**: Calculating the gradient over the entire dataset (Batch Gradient Descent) is computationally expensive and memory-intensive. In practice, we use **Stochastic Gradient Descent (SGD)** (one example at a time) or, more commonly, **Mini-batch Gradient Descent** (a small batch of examples). This introduces noise into the gradient estimate, which paradoxically helps the optimization process escape shallow local minima and saddle points.

## Optimization: Gradient Descent - Take 1

For the Cost Function $C(w, b)$, we need to find the minimum of this function with respect to the weights $w$ and biases $b$.

We calculate the gradient of the cost function with respect to each weight and bias using backpropagation 
$$
\nabla C = \begin{bmatrix}
\frac{\partial C}{\partial w} \\
\frac{\partial C}{\partial b}
\end{bmatrix}
$$

Then update the weights in the opposite direction of the gradient:

$$
w_{new} = w_{old} - \eta \frac{\partial C}{\partial w}
$$
$$
b_{new} = b_{old} - \eta \frac{\partial C}{\partial b}
$$

where $\eta$ is the **learning rate**. This update rule is called **Gradient Descent**.

### Why not Newton's Method?

You might ask, why not use faster optimization methods like **Newton's Method** (Newton-Raphson)?

Newton's method uses the second derivative (curvature) to find the minimum faster.
$$ w_{new} = w_{old} - \frac{C'(w)}{C''(w)} $$

However, for a neural network with millions of weights, calculating the second derivative (the Hessian matrix) is computationally infeasible. Gradient Descent, which uses only the first derivative, is much more scalable and efficient for Deep Learning.

## Take 2 - Deeper Dive


The Error function is a scalar function of the weights and biases. 

The loss (error) is a scalar function of all weights and biases.

In linear regression with MSE, the loss is a convex quadratic in the parameters, so optimization is well-behaved (a bowl-shaped surface)(e.g. see left in picture).

In deep learning, the loss becomes non-convex because it is the result of composing many nonlinear transformations. This creates a complex landscape with saddle points, flat regions, and multiple minima (e.g. see right in picture).

![costfunction]

**How will Gradient Descent work in this case - non convex function?**

Gradient descent does not attempt to find the global minimum, but rather follows the local slope of the cost function and converges to a local minimum or a flat region.

The cost function is differentiable almost everywhere*. At any point in parameter space, the gradient indicates the direction of steepest local increase, and moving in the opposite direction reduces the cost. During optimization, the algorithm may encounter local minima or saddle points.

(*The function is not differentiable at the point where the function is zero ex ReLU. This is not a problem in practice, as optimization algorithms handle such points using [subgradients](images/subgradient.png))

In practice, deep learning works well despite non-convexity, partly because modern networks have millions of parameters and their loss landscapes contain many saddle points and wide, flat minima rather than [poor isolated local minima](images/poorlocalminima.png).

Furthermore, we rarely use full-batch gradient descent. Instead, we use variants such as Stochastic Gradient Descent (SGD) or mini-batch gradient descent.

In these methods, gradients are computed using a single training example or a small batch of examples rather than the entire dataset. The resulting gradient is an average over the batch and serves as a noisy approximation of the true gradient. This stochasticity helps the optimizer escape saddle points and sharp minima, enabling effective training in practice.


## Backpropagation

**What does it mean to take the derivative of a scalar function with respect to vector-valued parameters?**

Finding the gradient is the job of Backpropagation. 

Backpropagation is based on Automatic Differentiation. Auto Diff is a fancy term to describe creating a computational graph and then differentiating it- via chain rule.

Chain rule is a basic rule of differentiation of composite functions.

In Neural networks each function is composed of vector functions.

What is a derivative of a vector function? This is something hard to explain in a simple way. It is a matrix of gradients.

This paper - The Simple Essence of Automatic Differentiation
Extended version Conal Elliott explains this in detail.
Quoting from it below.

---

### Jacobian Matrix 

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

Moreover, each of these situations has an accompanying chain rule, which says how to differentiate the composition of two functions. 
Where the scalar chain rule involves multiplying two scalar derivatives, the vector chain rule involves "multiplying" two matrices $A$ and $B$ (the Jacobians), defined as follows:

$$
(A \cdot B)_{ij} = \sum_{k=1}^m A_{ik} \cdot B_{kj}
$$

Since one can think of scalars as a special case of vectors, and scalar multiplication as a special case of matrix multiplication, perhaps we've reached the needed generality.

The derivative of a function $f: a \to b$ at some value in $a$ is thus not a number, vector, matrix, or higher-dimensional variant, but rather a linear map (also called "linear transformation") from $a$ to $b$, which we will write as $a \multimap b$. 

The numbers, vectors, matrices, etc. mentioned above are all different representations of linear maps; and the various forms of "multiplication" appearing in their associated chain rules are all implementations of linear map composition for those representations.

---

This above passage is taken from the paper on auto differentiation by Conal Elliott. Why I put it here is that it applies to our case of gradient descent as well. 

So the Loss function is a function of weight vectors $w$ and bias vectors $b$. 

$$
C(w, b)
$$

To minimize this, we need the derivative. But since our input is a vector (the weights) and our output is a scalar (the loss), what does the derivative look like?

Following Conal Elliott's definition, the derivative is not just a number, but a Linear Map. It is a function that tells us: 'If we nudge the weights by a tiny vector $\vec{v}$, how much will the Loss change?'

We represent this abstract linear map using a concrete list of numbers called the Gradient Vector $\nabla C$.

$$
\nabla C = \begin{bmatrix}
\frac{\partial C}{\partial w} \\
\frac{\partial C}{\partial b}
\end{bmatrix}
$$

This linear map tells us how any small change in the parameters will change the loss. Gradient descent exploits this first-order approximation by choosing the parameter update that most rapidly decreases the loss.

Mathematically this is more rigorous than the usual explanation of rolling down the gradient/slope which does not make sense in the context of a vector space.

**So what is Gradient Descent?**

 Gradient Descent takes the Gradient Vector found by Backprop and performs a vector subtraction in the weight space:
 
 $$w_{new} = w_{old} - \eta \cdot \nabla C$$

where $\eta$ is the learning rate.

**And how do we find the gradient vector $\nabla C$?**  via Backpropagation

## Backpropagation

Dont be confused by the name. It is just calculating the gradient vector $\nabla C$. for each layer.

Gradient descent then uses this gradient vector to update the weights in each layer.

These two work in unison to find the minimum of the loss function.


Lets first look at a neural network as a scalar composition of functions.

Imagine a simple neural network with 3 layers. It is essentially a composition of three functions:

A neural network is a composition of vector-valued functions, followed by a scalar-valued cost function:

$$
C = \text{Cost}(a_3) \\
a_3 = L_3(L_2(L_1(x)))
$$

Where $L_1$, $L_2$ and $L_3$ are the three layers of the network and  

Each layer is defined as:

$$
z_i = W_i a_{i-1} + b_i, \quad a_i = \sigma(z_i)
$$

And gradient descent is defined as:

$$w_{i_{new}} = w_{i_{old}} - \eta \cdot \partial C / \partial w_i$$

Problem is to find the partial derivative of the loss function with respect to the weights at each layer.

To calculate how a change in the first layer's weights ($w_1$) affects the final Cost ($C$), we have to trace the "path of influence" all the way through the network.

A nudge in $w_1$ changes the output of Layer 1. The change in Layer 1 changes the input to Layer 2. The change in Layer 2 changes the input to Layer 3. The change in Layer 3 changes the final Cost.

Mathematically, we multiply the derivatives (Linear Maps) of these links together:

We need to update weights of three layers

$$w_{1_{new}} = w_{1_{old}} - \eta \cdot \partial C / \partial w_1$$

$$w_{2_{new}} = w_{2_{old}} - \eta \cdot \partial C / \partial w_2$$

$$w_{3_{new}} = w_{3_{old}} - \eta \cdot \partial C / \partial w_3$$


And for that we need to find $ \partial C / \partial w_1 $, $ \partial C / \partial w_2 $, $ \partial C / \partial w_3 $.

Lets write down the chain rule for each layer:

$$\frac{\partial C}{\partial w_1} = \frac{\partial C}{\partial L_3} \cdot \frac{\partial L_3}{\partial L_2} \cdot \frac{\partial L_2}{\partial L_1} \cdot \frac{\partial L_1}{\partial w_1}$$

$$\frac{\partial C}{\partial w_2} = \frac{\partial C}{\partial L_3} \cdot \frac{\partial L_3}{\partial L_2} \cdot \frac{\partial L_2}{\partial w_2}$$

$$\frac{\partial C}{\partial w_3} = \frac{\partial C}{\partial L_3} \cdot \frac{\partial L_3}{\partial w_3}$$

Why is this written this way? By the chain rule, the derivative of a composition of functions is the product of the derivatives of the functions. It is thus easy to calculate the gradient of the loss with respect to the weights of each layer.

Lets calculate the gradient of the loss with respect to the weights of the first layer.

Notice something interesting?

*   To calculate $\frac{\partial C}{\partial w_3}$, we need $\frac{\partial C}{\partial L_3}$.

*   To calculate $\frac{\partial C}{\partial w_2}$, we need $\frac{\partial C}{\partial L_3} \cdot \frac{\partial L_3}{\partial L_2}$.

*   To calculate $\frac{\partial C}{\partial w_1}$, we need $\frac{\partial C}{\partial L_3} \cdot \frac{\partial L_3}{\partial L_2} \cdot \frac{\partial L_2}{\partial L_1}$.

We are re-calculating the same terms over and over again!

If we start from the **Output** (Layer 3) and move **Backwards**:
1.  We calculate $\frac{\partial C}{\partial L_3}$ once. We use it to find the update for $w_3$.

2.  We pass this value back to find $\frac{\partial C}{\partial L_2}$ (which is $\frac{\partial C}{\partial L_3} \cdot \frac{\partial L_3}{\partial L_2}$). We use it to find the update for $w_2$.

3.  We pass *that* value back to find $\frac{\partial C}{\partial L_1}$. We use it to find the update for $w_1$.

This avoids redundant calculations and is why it's called **Backpropagation**.

It is essentially **Dynamic Programming** applied to the Chain Rule.





### The Backpropagation Algorithm Step-by-Step


**Step 1: The Output Layer ($L_3$)**

We want to find the gradient $\frac{\partial C}{\partial w_3}$.
Using the Chain Rule:
$$ \frac{\partial C}{\partial w_3} = \frac{\partial C}{\partial a_3} \cdot \frac{\partial a_3}{\partial z_3} \cdot \frac{\partial z_3}{\partial w_3} $$

Let's break it down term by term:

1.  **Derivative of Cost w.r.t Activation** ($\frac{\partial C}{\partial a_3}$):
    For MSE $C = \frac{1}{2}(a_3 - y)^2$:
    $$ \frac{\partial C}{\partial a_3} = (a_3 - y) $$

2.  **Derivative of Activation w.r.t Input** ($\frac{\partial a_3}{\partial z_3}$):
    Since $a_3 = \sigma(z_3)$:
    $$ \frac{\partial a_3}{\partial z_3} = \sigma'(z_3) $$

3.  **Derivative of Input w.r.t Weights** ($\frac{\partial z_3}{\partial w_3}$):
    Since $z_3 = w_3 a_2 + b_3$:
    $$ \frac{\partial z_3}{\partial w_3} = a_2 $$

**Combining them:**
We define the "error" term $\delta_3$ at the output layer as:
$$ \delta_3 = \frac{\partial C}{\partial z_3} = (a_3 - y) \odot \sigma'(z_3) $$

> **Note on $\odot$ (Hadamard Product)**: We use element-wise multiplication here because both $(a_3 - y)$ and $\sigma'(z_3)$ are vectors of the same size.
>
> The Jacobian of an element-wise activation $\sigma$ is a diagonal matrix:
> $$ \frac{\partial a}{\partial z} = \text{diag}(\sigma'(z)) $$
>
> So multiplying by it is the same as a Hadamard product:
> $$ \text{diag}(\sigma'(z)) \, v = v \odot \sigma'(z) $$

So the gradient for the weights is:
$$ \frac{\partial C}{\partial w_3} = \delta_3 \cdot a_2^T $$

> **Note on Transpose ($a_2^T$)**: In backprop, we push gradients through a linear map $z = Wa + b$. The Jacobian w.r.t. $a$ is $W$, so the chain rule gives:
>
> $$ \frac{\partial C}{\partial a} = W^T \frac{\partial C}{\partial z} $$
>
> The transpose appears because we’re applying the transpose (adjoint) of the Jacobian to move gradients backward.

**Result**: We have the update for $w_3$.

$$w_{3_{new}} = w_{3_{old}} - \eta \cdot \partial C / \partial w_3$$


**Step 2: Propagate Back to $L_2$**

Now we need to find the gradient for the second layer weights: $\frac{\partial C}{\partial w_2}$.
Using the Chain Rule, we can reuse the error from the layer above:
$$ \frac{\partial C}{\partial w_2} = \frac{\partial C}{\partial z_2} \cdot \frac{\partial z_2}{\partial w_2} = \delta_2 \cdot a_1^T $$

But what is $\delta_2$ (the error at layer 2)?
$$ \delta_2 = \frac{\partial C}{\partial z_2} = \frac{\partial C}{\partial z_3} \cdot \frac{\partial z_3}{\partial z_2} $$

We know $\frac{\partial C}{\partial z_3} = \delta_3$.
And since $z_3 = w_3 \sigma(z_2) + b_3$:
$$ \frac{\partial z_3}{\partial z_2} = w_3 \cdot \sigma'(z_2) $$

So, we can calculate $\delta_2$ by "backpropagating" $\delta_3$:
$$ \delta_2 = (w_3^T \cdot \delta_3) \odot \sigma'(z_2) $$

**The Update Rule for Layer 2:**
$$ \frac{\partial C}{\partial w_2} = \delta_2 \cdot a_1^T $$

**Result**: We have the update for $w_2$.
$$w_{2_{new}} = w_{2_{old}} - \eta \cdot \frac{\partial C}{\partial w_2}$$

**Step 3: Propagate Back to $L_1$**

We repeat the exact same process to find the error at the first layer $\delta_1$.
$$ \delta_1 = (w_2^T \cdot \delta_2) \odot \sigma'(z_1) $$

**The Update Rule for Layer 1:**
$$ \frac{\partial C}{\partial w_1} = \delta_1 \cdot x^T $$
(Recall that $a_0 = x$, the input).

**Result**: We have the update for $w_1$.
$$w_{1_{new}} = w_{1_{old}} - \eta \cdot \frac{\partial C}{\partial w_1}$$

### Summary

So, Backpropagation is the efficient execution of the Chain Rule by utilizing the linear maps of each layer in reverse order.
*   It computes the local linear map (Jacobian) of a layer.
*   It takes the incoming gradient vector from the future layer.
*   It performs a Vector-Jacobian Product to pass the gradient to the past layer.




[neuralnetwork]: https://i.imgur.com/gE3QKCf.png
[costfunction]: images/costfunction.png