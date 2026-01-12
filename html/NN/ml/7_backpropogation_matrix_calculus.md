# The Mathematical Intuition Behind Deep Learning

Alex Punnen \
&copy; All Rights Reserved

---

[Contents](index.md)


# Chapter 6

## Back Propagation -Matrix Calculus

The previous chapters we used a Scalar derivation of the Back Propagation formula to implement it in a simple two layer neural network. What we have done is is to use Hadamard product and matrix transposes with scalar derivation alignment.

But we have not really explained why we use Hadamard product and matrix transposes with scalar derivation alignment.

This is due to Matrix Calculus which is the real way in which we should be deriving the Back Propagation formula.

Lets explore this in this chapter. Note that we are still not using a Softmax activation function in the output layer as is usually the case with Deep Neural Networks. Deriving the Back Propagation formula with Softmax activation function is bit more complex and we will do that in a later chapter.

Let's take the previous two layered simple neural network,with a Mean Square Error Loss function, and derive the Back Propagation formula with Matrix Calculus now.

Let's write the  equation of the following neural network

```python
x is the Input
y is the Output.
l is the number of layers of the Neural Network.
a is the activation function ,(we use sigmoid here)
```

$$
 x \rightarrow a^{l-1} \rightarrow  a^{l} \rightarrow  y
$$

Where the activation $a^l$ is
$$
  a^{l} = \sigma(w^l a^{l-1}+b^l).
$$

and

$$
a^{l} = \sigma(z^l) \quad where \quad
z^l =w^l a^{l-1} +b^l
$$

Our two layer neural network can be written as

 $$
 \mathbf { a^0 \rightarrow a^{1} \rightarrow  a^{2} \rightarrow  y }
 $$

($a^2$ does not denote the exponent but just that it is of layer 2)



## Some Math Intuition

The concept of a Neural Network as a composition of functions remains central.

In our network, most layers represent functions that map a **Vector to a Vector** ($\mathbb{R}^n \to \mathbb{R}^m$). For example, the hidden layers take an input vector and produce an activation vector.

However, the final step—calculating the Loss—is different. It maps the final output vector $a^L$ (and the target $y$) to a single **Scalar** value, the Cost $C$ ($\mathbb{R}^n \to \mathbb{R}$).

### Gradient Vector

When we take the derivative of a scalar-valued function (like the Cost $C$) with respect to a vector (like the weights $w$), the result is a vector of the same size as $w$. This is called the **Gradient Vector**.

$$
\nabla_w C = \begin{bmatrix} \frac{\partial C}{\partial w_1} \\ \vdots \\ \frac{\partial C}{\partial w_n} \end{bmatrix}
$$

Why is it called a “gradient”?

Because the gradient points in the direction of steepest increase of the function.

Moving a tiny step along $+\nabla_w C$ increases the cost the fastest.

Moving a tiny step against it, i.e. along $-\nabla_w C$, decreases the cost the fastest.

That’s exactly why gradient descent updates parameters like this:

$$
w \leftarrow w - \eta \nabla_w C
$$

where $\eta$ is the learning rate.

So the gradient vector is more than a list of derivatives—it’s the local direction that tells us how to change parameters to reduce the loss.

See the Colab [1] for a generated visualization of this.

The first image is the plotting of the Cost function

![gradient vector](images/gradientvec1.png)

The second image where you see the cones are the gradient vector of the Cost function wrto weights plotted in 3D space.

![gradient vector](images/gradientvec2.png)


## Jacobian Matrix

The second key concept is the **Jacobian Matrix**.

As mentioned earlier, in our network, most layers represent functions that map a **Vector to a Vector** ($\mathbb{R}^n \to \mathbb{R}^m$). For example, a hidden layer takes an input vector $x$ and produces an activation vector $a$.

What is the derivative of a vector-valued function with respect to a vector input? This is where the Jacobian comes in.

For a function $f: \mathbb{R}^n \to \mathbb{R}^m$ that maps an input vector $x$ of size $n$ to an output vector $y$ of size $m$, the derivative is an $m \times n$ matrix called the Jacobian Matrix $J$.

The entry $J_{ij}$ is the partial derivative of the $i$-th output component with respect to the $j$-th input component:

$$
J_{ij} = \frac{\partial f_i}{\partial x_j}
$$

### The Chain Rule with Matrices

The beauty of the Jacobian is that it allows us to generalize the chain rule.

For scalar functions, the chain rule is just multiplication: $(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$.

For vector functions, the chain rule becomes **Matrix Multiplication** of the Jacobians:

If we have a composition of functions $y = f(g(x))$, and we let $A$ be the Jacobian of $f$ and $B$ be the Jacobian of $g$, then the Jacobian of the composition is simply the matrix product $A \cdot B$.

$$
(A \cdot B)_{ij} = \sum_{k=1}^m A_{ik} \cdot B_{kj}
$$

So, the Jacobian is a matrix of partial derivatives that represents the local linear approximation of a vector function. When we say the Jacobian represents a "local linear approximation," we mean:

$$ \text{Change in Output} \approx \text{Jacobian Matrix} \cdot \text{Change in Input} $$

$$ \Delta y \approx J \cdot \Delta x $$

It tells us: "If I nudge the input vector by a tiny vector $\Delta x$, the output vector will change by roughly the matrix-vector product $J \cdot \Delta x$."

### BackPropogation Trick - VJP (Vector Jacobian Product) and JVP (Jacobian Vector Product) 

There is one more trick that we can use to make backpropogation more efficient. 

Let me explain with an example.

Suppose we have a chain of functions: $y = f(g(h(x)))$.
To find the derivative $\frac{\partial y}{\partial x}$, the chain rule tells us to multiply the Jacobians:
$$ J_{total} = J_f \cdot J_g \cdot J_h $$

If $x, h, g, f$ are all vectors of size 1000, then each Jacobian is a $1000 \times 1000$ matrix. Multiplying them is expensive ($O(N^3)$).

**However, in Backpropagation, we always start with a scalar Loss function.**
The final derivative $\frac{\partial C}{\partial y}$ is a row vector (size $1 \times N$).

So we are computing:
$$ \nabla C = \underbrace{\frac{\partial C}{\partial y}}_{1 \times N} \cdot \underbrace{J_f}_{N \times N} \cdot \underbrace{J_g}_{N \times N} \cdot \underbrace{J_h}_{N \times N} $$

Notice the order of operations matters!
1.  **Jacobian-Matrix Product**: If we multiply the matrices first ($J_f \cdot J_g$), we do expensive matrix-matrix multiplication.
2.  **Vector-Jacobian Product (VJP)**: If we multiply from left to right:
    *   $v_1 = \frac{\partial C}{\partial y} \cdot J_f$ (Vector $\times$ Matrix $\to$ Vector)
    *   $v_2 = v_1 \cdot J_g$ (Vector $\times$ Matrix $\to$ Vector)
    *   $v_3 = v_2 \cdot J_h$ (Vector $\times$ Matrix $\to$ Vector)

We **never** explicitly compute or store the full Jacobian matrix. We only compute the product of a vector with the Jacobian. This is much faster ($O(N^2)$) and uses less memory.

This is the secret sauce of efficient Backpropagation! 


---


## Gradient Vector/2D-Tensor of Loss function in last layer

$$
C = \frac{1}{2} \sum_j (y_j-a^L_j)^2
$$

Assuming a neural net with 2 layers, we have the final Loss as 

$$
C = \frac{1}{2} \sum_j (y_j-a^2_j)^2
$$

Where

$$
a^2 = \sigma(w^2.a^1)
$$

We can then write

$$
C = \frac{1}{2} \sum_j v^2 \quad \rightarrow (Eq \;A)
$$

Where

$$
v= y-a^2
$$

## Partial Derivative of Loss function with respect to Weight

For the last layer, lets use Chain Rule to split like below

$$
\frac {\partial C}{\partial w^2} = \frac{\partial v^2}{\partial v} \cdot \frac{\partial v}{\partial w^2} \quad \rightarrow (Eq \;B)
$$

$$
 \frac{\partial v^2}{\partial v} =2v \quad \rightarrow (Eq \;B.1)
$$

$$
\frac{\partial v}{\partial w^2}=  \frac{\partial (y-a^2)}{\partial w^2} = 
0-\frac{\partial a^2}{\partial w^2} \quad \rightarrow (Eq \;B.2)
$$

$$
\frac {\partial C}{\partial w^2} = \frac{1}{2} \cdot 2v(0-\frac{\partial a^2}{\partial w^2}) \quad \rightarrow (Eq \;B)
$$
&nbsp;

### Now we need to find $$\frac{\partial a^2}{\partial w^2}$$

Let

$$
\begin{aligned}
a^2= \sigma(\sum(w^2 \otimes a^1 )) = \sigma(z^2) 
\\\\
z^2 =  \sum(w^2 \otimes a^1)
\\\\
z^2 = \sum(k^2) \; \text {, where} \; k^2=w^2 \odot a^1 
\end{aligned}
$$

We now need to derive an intermediate term which we will use later

$$
\begin{aligned}
\frac{\partial z^2}{\partial w^2} =\frac{\partial z^2}{\partial k^2}*\frac{\partial k^2}{\partial w^2}
\\\\
=\frac {\partial \sum(k^2)}{\partial k^2} \cdot \frac {\partial (w^2 \odot a^1 )} {\partial w^2}
\\ \\
\frac{\partial z^2}{\partial w^2} = (\vec{1})^T \cdot \text{diag}(a^1) =(a^{1})^T \quad \rightarrow (Eq \;B.3)
\end{aligned}
$$

Though these are written like scalar here; actually all these are partial differentiation of Vector by Vector, or Vector by Scalar. A set of vectors can be represented as the matrix here.More details here https://explained.ai/matrix-calculus/#sec6.2


The Vector dot product $w.a$ when applied on matrices becomes the sum of elementwise multiplication (also called Hadamard product) $\sum w^2 \otimes a^1$ 

Going back to  $Eq \;(B.2)$

$$
\frac {\partial a^2}{\partial w^2} = \frac{\partial a^2}{\partial z^2} \cdot \frac{\partial z^2}{\partial w^2}
$$

Using $Eq \;(B.3)$ for the term in left

$$
=  \frac{\partial a^2}{\partial z^2} \cdot (a^{1})^T
$$

$$
=  \frac{\partial \sigma(z^2)}{\partial z^2} \cdot (a^{1})^T
$$

$$
\frac {\partial a^2}{\partial w^2} =   \sigma^{'}(z^2) \cdot (a^{1})^T \quad \rightarrow (Eq \;B.4)
$$

Now lets got back to partial derivative of Loss function wrto to weight

$$
\frac {\partial C}{\partial w^2} = \frac {1}{2} \cdot 2v(0-\frac{\partial a^2}{\partial w^2}) \quad \rightarrow (Eq \;B)
$$

Using $Eq \;(B.4)$ to substitute in the last term

$$
\begin{aligned}
= v(0- \sigma^{'}(z^2) * (a^{1})^T) 
\\\\
= v*-1*\sigma^{'}(z^2) * (a^{1})^T
\\\\
= (y-a^2)*-1*\sigma^{'}(z^2) * (a^{1})^T
\\\\
\frac {\partial C}{\partial w^2}= (a^2-y) \cdot \sigma^{'}(z^2) \cdot (a^{1})^T \quad \rightarrow Eq \; (3)
\end{aligned}
$$
&nbsp;

## Gradient Vector of Loss function in Inner Layer

---

&nbsp;

Now let's do the same for the inner layer. This is bit more tricky and we use the Chain rule to derive this

&nbsp;

$$
\frac {\partial C}{\partial w^1} = \frac {\partial a^1}{\partial w^1} \cdot \frac {\partial C}{\partial a^1}  \quad \rightarrow (4.0)
$$

&nbsp;

We can calculate the first part of this from $Eq\; (B.4)$ that we derived above

$$
\begin{aligned}
\frac {\partial a^2}{\partial w^2} =   \sigma^{'}(z^2) \cdot (a^{1})^T \quad \rightarrow (Eq \;B.4)
\\\\
\frac {\partial a^1}{\partial w^1}  = \sigma'(z^1) \cdot (a^{0})^T \quad \rightarrow (4.1)
\end{aligned}
$$


For the second part, we use Chain Rule to split like below, the first part of which we calculated in the earlier step.

$$
\begin{aligned}
\frac{\partial C}{\partial(a^1)} =  \frac{\partial C}{\partial(a^2)}.\frac{\partial(a^2)}{\partial(a^1)}
\\\\
{
\frac{\partial C}{\partial(a^2)} = \frac {\partial({\frac{1}{2} \|y-a^2\|^2)}}{\partial(a^2)} = \frac{1}{2} \cdot 2 \cdot (a^2-y) =(a^2-y) = \delta^{2}  }
\\\\
\frac {\partial C}{\partial(a^2)}  =\delta^{2}  \rightarrow (4.2)\\ \\

\text {Now to calculate} \quad

 \frac{\partial(a^2)}{\partial(a^1)} \quad where \quad

a^{2} = \sigma(w^2 a^{1}+b^2) \\ \\

\frac{\partial(a^2)}{\partial(a^1)} = \frac{\partial(\sigma(w^2 a^{1}+b^2))}{\partial(a^1)} =  w^2 \cdot \sigma'(w^2 a^{1}+b^2) = w^2 \cdot \sigma'(z^2)\rightarrow (4.3)*

\end{aligned}
$$

*<https://math.stackexchange.com/a/4065766/284422>

Putting (4.1) (4.2) and (4.3) together

## Final Equations

$$  \mathbf{
\frac {\partial C}{\partial w^1} = \sigma'(z^1) \cdot (a^{0})^T \cdot \delta^{2} \cdot w^2 \cdot \sigma'(z^2) \quad \rightarrow Eq \; (5)
}$$

$$
\delta^2 = (a^2-y)
$$

Adding also the partial derivate of loss funciton with respect to weight in the final layer

$$ \mathbf{
\frac {\partial C}{\partial w^2}= \delta^{2} \cdot \sigma^{'}(z^2) \cdot (a^{1})^T \quad \rightarrow Eq \; (3)
}
$$

You can see that the inner layer derivative have terms from the outer layer. So if we store and use the result; this is like dynamic program; maybe the back-propagation algorithm is the most elegant dynamic programming till date.

$$  \mathbf{
\frac {\partial C}{\partial w^1} = \delta^{2} \cdot \sigma'(z^2) \cdot (a^{0})^T \cdot w^2 \cdot \sigma'(z^1) \quad \rightarrow Eq \; (5)
}$$

&nbsp;

## Using Gradient Descent to find the optimal weights to reduce the Loss function

&nbsp;

With equations (3) and (5) we can calculate the gradient of the Loss function with respect to weights in any layer - in this example 

$$\frac {\partial C}{\partial w^1},\frac {\partial C}{\partial w^2}$$

&nbsp;

We now need to adjust the previous weight, by gradient descent.

&nbsp;

So using the above gradients we get the new weights iteratively like below. If you notice this is exactly what is happening in gradient descent as well; only chain rule is used to calculate the gradients here. Backpropagation is the algorithm that helps calculate the gradients for each layer.

&nbsp;

$$\mathbf {
  W^{l-1}_{new} = W^{l-1}_{old} - \eta \cdot \frac{\partial C}{\partial w^{l-1}}
}$$


&nbsp;

Where $\eta$ is the learning rate.

Reference  

- <https://cedar.buffalo.edu/~srihari/CSE574/Chap5/Chap5.3-BackProp.pdf>
- <http://neuralnetworksanddeeplearning.com/chap2.html>

 [1]: https://colab.research.google.com/drive/1sMODrDCdR7lKF9cWcNNhhdLglxJRzmgK?usp=sharing

Next: [Back Propagation in Full - With Softmax & CrossEntropy Loss](8_backpropogation_full.md)