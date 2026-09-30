# Gradient Descent

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
  a^{l} = \sigma(W^l a^{l-1} + b^l)
$$

where $a^0 = x$ (the input).

The linear transformation:

$z = w^T x + b$ 

defines a hyperplane (decision boundary) in the feature space.

The activation function then introduces *non-linearity*, allowing the network to combine **multiple such hyperplanes into complex decision boundaries.**

#### Why Non-Linearity Is Non-Negotiable

Without activation:

$$
f(x) = W^{L} W^{L-1} \dots W^{1} x
$$

This collapses to:

$$
f(x) = Wx
$$

Still one big linear transformation and hence one hyperplane; the problems of not able to separate features will come. Only because of non-linearity, we can get  multiple hyperplanes and hence a composable complex decision boundaries that can separate features.


So the concept of Vectors, Matrices and Hyperplanes remain the same as before. Let us explore the chain of functions part here

A neural network with $L$ layers can be represented as a nested function:$$f(x) = f^{L}(...f^{2}(f^{1}(x))...)$$

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

1. It does not attempt to find the **global minimum**, but rather follows the **local slope** of the cost function and converges to a critical point: a local minimum, a saddle point, or a flat region. A **saddle point** is a critical point where the gradient is zero but the point is not a minimum; in high dimensions such points are common, and an optimizer must escape them rather than settle on them.

2. Gradients can **vanish or explode**, leading to slow or unstable convergence. The practical solution to control this is to use **learning rate** and using **adaptive learning rate** methods like **Adam** or **RMSprop**.

3. **Batch Size matters**: Calculating the gradient over the entire dataset (Batch Gradient Descent) is computationally expensive and memory-intensive. In practice, we use **Stochastic Gradient Descent (SGD)** (one example at a time) or, more commonly, **Mini-batch Gradient Descent** (a small batch of examples). This introduces noise into the gradient estimate, which paradoxically helps the optimization process escape shallow local minima and saddle points.

## Optimization: Gradient Descent — Take 1

Gradient Descent is a simple yet powerful optimization algorithm used to minimize functions by iteratively updating parameters in the direction that reduces the function's output.

For basic scalar functions (e.g., $( f(x) = x^2 )$), the update rule is straightforward:
$$
x \leftarrow x - \eta \frac{df}{dx}
$$
where $( \eta )$ is the learning rate.

However, **neural networks are not simple scalar functions**. They are **composite vector-valued functions** — layers of transformations that take in high-dimensional input vectors and eventually output either vectors (like logits) or scalars (like loss values).

Understanding how to optimize these complex, high-dimensional functions requires us to extend basic calculus:
- The **gradient vector** helps when the function outputs a scalar but takes a vector input (e.g., a loss function w.r.t. weights).
- The **Jacobian matrix** becomes important when both the input and the output are vectors (e.g., when computing gradients layer by layer in backpropagation).

We'll build up to this step by step — starting with scalar gradients, then moving to vector calculus, Jacobians, and how backpropagation stitches it all together.

Let’s take it one layer at a time.


## Gradient Descent for Scalar Functions

Consider this simple system that composes two functions:

$$o = g(f(x, w^1), w^2)$$

Where:
- $x$ is your input (fixed, given by your data)
- $w^1$ and $w^2$ are **parameters you can adjust** (like weights in a neural network)
- $f$ is the first function (think: first layer)
- $g$ is the second function (think: second layer)
- $o$ is the final output

  

Let's make this concrete with simple linear functions:

$$f(x, w^1) = x \cdot w^1 + b^1$$
$$g(z, w^2) = z \cdot w^2 + b^2$$

So the full composition is:

$$o = g(f(x, w^1), w^2) = (x \cdot w^1 + b^1) \cdot w^2 + b^2$$

  

### Running the Numbers: A Real Example

Let's pick actual values and see what happens:

**Fixed values:**
- Input: $x = 2.0$
- Bias terms: $b^1 = 1.0$, $b^2 = 0.5$

**Current parameter values:**
- $w^1 = 0.5$
- $w^2 = 1.5$

**Step 1**: Compute intermediate result from first function:

$$z = f(x, w^1) = 2.0  \times  0.5 + 1.0 = 2.0$$

**Step 2**: Compute final output from second function:

$$o = g(z, w^2) = 2.0  \times  1.5 + 0.5 = 3.5$$

  

**The problem**: Suppose we want $o_{\text{target}} = 5.0$ instead!

  

Our current error is:

  

$$C = \frac{1}{2}(o - o_{\text{target}})^2 = \frac{1}{2}(3.5 - 5.0)^2 = \frac{1}{2}(-1.5)^2 = 1.125$$

  

**The million-dollar question**: How should we change $w^1$ and $w^2$ to reduce this error?

  

### The Adjustment Problem: Which Direction? How Much?

Here's what we need to know:

1.  **Should we increase or decrease $w^1$?** (Which direction?)
2.  **How sensitive is $o$ to changes in $w^1$?** (How much?)
3.  **Same questions for $w^2$.**

  

This is where derivatives come in! Specifically, we need:

  

$$\frac{\partial o}{\partial w^1} \quad  \text{and} \quad  \frac{\partial o}{\partial w^2}$$

  

These tell us:

-  **Sign**: Positive means "increase $w$ increases $o$", negative means the opposite

-  **Magnitude**: Larger absolute value means $o$ is more sensitive to changes in $w$

But there's a complication: $w^1$ doesn't directly affect $o$. It affects $f$, which then affects $g$, which then affects $o$. This is a **composition**, and we need to trace the effect through multiple steps.

This is where the "Chain Rule" of Calculus comes into play.

### The Chain of Effects

Let's visualize how changes propagate:

```
Change w¹ → Affects f → Changes z → Affects g → Changes o
      ↓            ↓         ↓               ↓
      Δw¹        ∂f/∂w¹     Δz     ∂g/∂z     Δo
```

Similarly for $w^2$ (but $w^2$ directly affects $g$):

```
Change w² → Affects g → Changes o
↓ ↓ ↓
Δw² ∂g/∂w² Δo
```

  

The key insight: **To find how $w^1$ affects $o$, we need to multiply the effects at each step.**

  

This is the **chain rule** in action!

  

### The Solution: Applying the Chain Rule

  

For our composition $o = g(f(x, w^1), w^2)$, let's introduce a shorthand: call $z = f(x, w^1)$ the intermediate value.

  

Then:

$$o = g(z, w^2)$$

  

**Computing $\frac{\partial o}{\partial w^1}$:**

By the chain rule of calculus:

$$\frac{\partial o}{\partial w^1} = \frac{\partial o}{\partial z} \cdot  \frac{\partial z}{\partial w^1}$$

Let's compute each piece:

**Part 1**: How does $o$ change with $z$?

$$\frac{\partial o}{\partial z} = \frac{\partial}{\partial z}(z \cdot w^2 + b^2) = w^2 = 1.5$$

**Part 2**: How does $z$ change with $w^1$?

$$\frac{\partial z}{\partial w^1} = \frac{\partial}{\partial w^1}(x \cdot w^1 + b^1) = x = 2.0$$

**Putting it together**:

$$\frac{\partial o}{\partial w^1} = 1.5  \times  2.0 = 3.0$$

**Interpretation**: If we increase $w^1$ by 0.1, then $o$ increases by approximately $3.0  \times  0.1 = 0.3$.

**Computing $\frac{\partial o}{\partial w^2}$:**

This is simpler because $w^2$ directly affects $g$:

$$\frac{\partial o}{\partial w^2} = \frac{\partial}{\partial w^2}(z \cdot w^2 + b^2) = z = 2.0$$

**Interpretation**: If we increase $w^2$ by 0.1, then $o$ increases by approximately $2.0  \times  0.1 = 0.2$.

### Making the Update: Gradient Descent

  

Now we can adjust our parameters! Since we want to **increase** $o$ from 3.5 to 5.0, and both gradients are positive, we should increase both $w^1$ and $w^2$.

Using gradient descent with learning rate $\eta = 0.2$:

$$(w^1)^{\text{new}} = w^1 + \eta  \cdot  \frac{\partial o}{\partial w^1} = 0.5 + 0.2  \times  3.0 = 0.5 + 0.6 = 1.1$$

$$(w^2)^{\text{new}} = w^2 + \eta  \cdot  \frac{\partial o}{\partial w^2} = 1.5 + 0.2  \times  2.0 = 1.5 + 0.4 = 1.9$$

**Note**: We're adding (not subtracting) because we want to increase $o$. Normally in machine learning, we minimize error, so we'd use $w - \eta  \cdot  \frac{\partial C}{\partial w}$.

  

### Verification: Did It Work?

  

Let's recompute with the new weights:

**Step 1**: New intermediate value:

$$z^{\text{new}} = x \cdot (w^1)^{\text{new}} + b^1 = 2.0  \times  1.1 + 1.0 = 3.2$$

**Step 2**: New output:

$$o^{\text{new}} = z^{\text{new}} \cdot (w^2)^{\text{new}} + b^2 = 3.2  \times  1.9 + 0.5 = 6.58$$

**Progress check**:
- Before: $o = 3.5$ (error from target = 1.5)
- After: $o = 6.58$ (error from target = -1.58)
- We overshot! But that's okay - we moved in the right direction

With a smaller learning rate (say $\eta = 0.1$), we'd get:
- $(w^1)^{\text{new}} = 0.8$, $(w^2)^{\text{new}} = 1.7$
- $z^{\text{new}} = 2.6$, $o^{\text{new}} = 4.92$
- Much closer to our target of 5.0!

    
This is how Gradient Descent works in a nutshell. The same concepts carry over in deep learning with some added complexity.

## Gradient Descent for a Two-Layer Neural Network (Scalar Form)

Let's apply this to a simple neural network with one hidden layer.
We have:
*   **Input**: $x$
*   **Hidden Layer**: 1 neuron with weight $w^1$, bias $b^1$, activation $\sigma$
*   **Output Layer**: 1 neuron with weight $w^2$, bias $b^2$, activation $\sigma$
*   **Target**: $y$

**Forward Pass:**
1.  $z^1 = w^1 x + b^1$
2.  $a^1 = \sigma(z^1)$
3.  $z^2 = w^2 a^1 + b^2$
4.  $a^2 = \sigma(z^2)$ (This is our prediction $\hat{y}$)

**Loss Function:**
We use the Mean Squared Error (MSE) for this single example:
$$ C = \frac{1}{2}(y - a^2)^2 $$

**Goal:**
Find $\frac{\partial C}{\partial w^1}, \frac{\partial C}{\partial b^1}, \frac{\partial C}{\partial w^2}, \frac{\partial C}{\partial b^2}$ to update the weights.

**Backward Pass (Deriving Gradients):**

**Layer 2 (Output Layer):**
We want how $C$ changes with $w^2$.
$$ \frac{\partial C}{\partial w^2} = \frac{\partial C}{\partial a^2} \cdot \frac{\partial a^2}{\partial z^2} \cdot \frac{\partial z^2}{\partial w^2} $$

*   $\frac{\partial C}{\partial a^2} = (a^2 - y)$ (Derivative of $\frac{1}{2}(y-a)^2$)
*   $\frac{\partial a^2}{\partial z^2} = \sigma'(z^2)$ (Derivative of activation)
*   $\frac{\partial z^2}{\partial w^2} = a^1$

So,
$$ \frac{\partial C}{\partial w^2} = (a^2 - y) \sigma'(z^2) a^1 $$

Let's define the "error term" for layer 2 as $\delta^2 = (a^2 - y) \sigma'(z^2)$.
Then:
$$ \frac{\partial C}{\partial w^2} = \delta^2 a^1 $$
$$ \frac{\partial C}{\partial b^2} = \delta^2 \cdot 1 = \delta^2 $$

**Layer 1 (Hidden Layer):**
We want how $C$ changes with $w^1$. The path is longer: $w^1 \to z^1 \to a^1 \to z^2 \to a^2 \to C$.
$$ \frac{\partial C}{\partial w^1} = \underbrace{\frac{\partial C}{\partial a^2} \cdot \frac{\partial a^2}{\partial z^2}}_{\delta^2} \cdot \frac{\partial z^2}{\partial a^1} \cdot \frac{\partial a^1}{\partial z^1} \cdot \frac{\partial z^1}{\partial w^1} $$

*   We know the first part is $\delta^2$.
*   $\frac{\partial z^2}{\partial a^1} = w^2$
*   $\frac{\partial a^1}{\partial z^1} = \sigma'(z^1)$
*   $\frac{\partial z^1}{\partial w^1} = x$

So,
$$ \frac{\partial C}{\partial w^1} = \delta^2 \cdot w^2 \cdot \sigma'(z^1) \cdot x $$

Let's define the error term for layer 1 as $\delta^1 = \delta^2 w^2 \sigma'(z^1)$.
Then:
$$ \frac{\partial C}{\partial w^1} = \delta^1 x $$
$$ \frac{\partial C}{\partial b^1} = \delta^1 $$

**The Update:**
$$ w^1 \leftarrow w^1 - \eta \delta^1 x $$
$$ w^2 \leftarrow w^2 - \eta \delta^2 a^1 $$

This pattern—calculating an error term $\delta$ at the output and propagating it back using the weights—is why it's called **Backpropagation**.

Note that we are using here scalar form of gradient descent and not directly applicable to real neural networks.  But this gives us the intuition of how backpropagation works.

## Some other notes related to Gradient Descent



The Loss/Cost function is a scalar function of the weights and biases. 

The loss/error is a scalar function of all weights and biases.

In simpler Machine Learning problems like linear regression with MSE, the loss is a convex quadratic in the parameters, so optimization is well-behaved (a bowl-shaped surface)(e.g. see left in picture).

In deep learning, the loss becomes non-convex because it is the result of composing many nonlinear transformations. This creates a complex landscape with saddle points, flat regions, and multiple minima (e.g. see right in picture).

![costfunction]

**How will Gradient Descent work in this case - non convex function?**

Gradient descent does not attempt to find the **global minimum**, but rather follows the local slope of the cost function and converges to a local minimum or a flat region.

The Loss function is differentiable almost everywhere*. At any point in parameter space, the gradient indicates the direction of steepest local increase, and moving in the opposite direction reduces the cost. During optimization, the algorithm may encounter local minima or saddle points.

(*The function is not differentiable at the point where the function is zero ex ReLU. This is not a problem in practice, as optimization algorithms handle such points using [subgradients](images/subgradient.png))

In practice, deep learning works well despite non-convexity, partly because modern networks have millions of parameters and their loss landscapes contain many saddle points and wide, flat minima rather than [poor isolated local minima](images/poorlocalminima.png).

Also we rarely use full-batch gradient descent. Instead, we use variants such as Stochastic Gradient Descent (SGD) or mini-batch gradient descent that acts as form of sampling.

In these methods, gradients are computed using a single training example or a small batch of examples rather than the entire dataset. 

The resulting gradient is an average over the batch and serves as a noisy approximation of the true gradient. This stochasticity helps the optimizer escape saddle points and sharp minima, enabling effective training in practice.


Next: [Backpropagation with Scalar Calculus](4_backpropogation_chainrule.md)



## References

- [Neural Networks and Deep Learning - Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap2.html)
- [Gradient Descent - Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)

[neuralnetwork]:  images/neuralnet2.png
[costfunction]: images/costfunction.png
