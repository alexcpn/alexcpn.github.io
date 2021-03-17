# Chapter 5: Back Propagation for a Two layered Neural Network

Let's take the simple neural network and walk through the same, first going through the maths and then the implementation.

Let's write the  equation of the following neural network

```
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
http://neuralnetworksanddeeplearning.com/chap2.html#eqtn25

and
$$
a^{l} = \sigma(z^l) \quad where \quad
z^l =w^l a^{l-1} +b^l
$$
<a name="(a)"></a>
$$
\text{Also lets write the derivate}\;  \; 
\mathbf {\frac{\partial a^{l} }{\partial w^{l}} = \frac{\partial \sigma (z^{l}) }{\partial w^{l}} = \sigma' (z^{l}) \quad \rightarrow  ( {a})}
$$

Where $\sigma'$ = derivative of Sigmoid with respect to Weight at layer l



\
&nbsp;

# A Two Layered Neural Network

Let's start with a concrete case of a Neural network with two layers and derive the equations of back propagation for that first. Each of these are explained in more detail in the previous sections.

&nbsp;

Our two layer neural network can be written as

 $$
 \mathbf { a^0 \rightarrow a^{1} \rightarrow  a^{2} \rightarrow  y }
 $$

($a^2$ does not denote the exponent but just that it is of layer 2)

Lets write down the derivative of Loss function wrto weight using chain rule

$$
\mathbf {
\frac {\partial C}{\partial w^l} 
= \frac {\partial a^l}{\partial w^l} . \frac {\partial C}{\partial a^l}
}
$$

We will use the above equation as the basis for the rest of the chapter.

&nbsp;

## Gradient Vector of Loss function In Output Layer
---
&nbsp;

Our Loss function is $C$. Lets substitute $l$  with layer numbers and get the gradient of the Cost with respect to weights in layer 2 and layer 1.

$$
\frac {\partial C}{\partial w^2}= \frac {\partial a^2}{\partial w^2}.  \frac {\partial C}{\partial a^2}
$$

The first term follows from [Equation (a)](#(a))

$$
\mathbb{
\frac{\partial a^{2} }{\partial w^2} = \sigma' (z^{2}) \quad \rightarrow  (\mathbf  {1})
}
$$

To find the second term, we do normal derivation

<a name="Eq2"></a>

$$
\mathbf{
\frac{\partial C}{\partial(a^2)} = \frac {\partial({\frac{1}{2} \|y-a^2\|^2)}}{\partial(a^2)} = \frac{1}{2}*2*(a^2-y) =(a^2-y) = \delta^{2} \rightarrow (2) }
$$

Putting 1 & 2 together we get the final equation for the second layer. This is the output layer.

&nbsp;

$$ \mathbf{
\frac {\partial C}{\partial w^2} = \sigma'(z^{2})*(a^2-y) =\sigma'(z^{2})*\delta{^2} \quad \rightarrow (3) }
$$
&nbsp;
http://neuralnetworksanddeeplearning.com/chap2.html#eqtnBP1

&nbsp;

## Gradient Vector of Loss function in Inner Layer
---

&nbsp;

Now let's do the same for the inner layer. This is bit more tricky and we use the Chain rule to derive this

&nbsp;

$$
\frac {\partial C}{\partial w^1} = \frac {\partial a^1}{\partial w^1} . \frac {\partial C}{\partial a^1}  \quad \rightarrow (4.0)
$$

&nbsp;

We can calculate the first part of this like below

$$
\frac {\partial a^1}{\partial w^1}  = \sigma'(z^1) \quad \rightarrow (4.1)
$$

The first term follows from [Equation (a)](#(a))

For the second part, we use Chain Rule to split like below, the first part of which we calculated in the earlier step.

$$
\frac{\partial C}{\partial(a^1)} =  \frac{\partial C}{\partial(a^2)}.\frac{\partial(a^2)}{\partial(a^1)}
$$

Note that [previously](#Eq2) we  had calculated

$$\begin{aligned}

\frac {\partial C}{\partial(a^2)}  =\delta^{2}  \rightarrow (4.2)\\ \\

\text {Now to calculate} \quad

 \frac{\partial(a^2)}{\partial(a^1)} \quad where \quad

a^{2} = \sigma(w^2 a^{1}+b^2) \\ \\

\frac{\partial(a^2)}{\partial(a^1)} = \frac{\partial(\sigma(w^2 a^{1}+b^2))}{\partial(a^1)} =  w^2.\sigma'(w^2 a^{1}+b^2) = w^2.\sigma'(z^2)\rightarrow (4.3)\\ \\

Putting \space (4.1) \space  \space (4.2)\space  and (4.3)\space together \\ \\

\end{aligned}$$

(4.3) https://math.stackexchange.com/a/4065766/284422

---

$$
\mathbf{
\frac {\partial C}{\partial w^1} = \sigma'(z^1)*\delta^{2}*w^2 . \sigma'(z^2)\quad \rightarrow \mathbb (5)
}
$$
We substitute the first term equation (2).
Repeating here the previous equation (3) as well

$$ \mathbf{
\frac {\partial C}{\partial w^2} =\sigma'(z^{2})*\delta^{2} \quad \rightarrow (3) }
$$

Note that weight is a Vector and we need to use the Vector product /dot product where weights are concerned. We will do an implementation to test out these equations to be sure.

\
&nbsp;

----

&nbsp;

With equations (3) and (5) we can calculate the gradient of the Loss function with respect to weights in any layel - in this example $\frac {\partial C}{\partial w^1},\frac {\partial C}{\partial w^2}$

&nbsp;

We now need to adjust the previous weight, by gradient descent.

&nbsp;

So using the above gradients we get the new weights iteratively like below. If you notice this is exactly what is happening in gradient descent as well; only chain rule is used to calculate the gradients here. Backpropagation is the algorithm that helps calculate the gradients for each layer.

&nbsp;

$$
  W^{l-1}_{new} = W^{l-1}_{old} - learningRate* \delta C_0/ \delta w^{l-1}
$$

\
&nbsp;

# Implementation

With this clear, this is not so difficult to code up. Let's do this. I am following the blog and code here http://iamtrask.github.io/2015/07/12/basic-python-network/ adding little more explanation for each of the steps, from what we have learned.

We will use matrices to represent input and weight matrices.

```python
x = np.array(
    [
        [0,0,1],
        [0,1,1],
        [1,0,1],
        [1,1,1]
    ])

```

This is a 4*3 matrix. Note that each row is an input. lets take all this 4 as 'training set'

```python
y = np.array(
  [
      [0],
      [1],
      [1],
      [0]
  ])
```

This is a 4*1 matrix that represent the expected output. That is for input [0,0,1] the output is [0] and for [0,1,1] the output is [1] etc.

**A neural network is implemented as a set of matrices representing the weights of the network.**

Let's create a two layered network. Before that please not the formula for the neural network

So basically the output at layer l is the dot product of the weight matrix of layer l and input of the previous layer.

Now let's see how the matrix dot product works based on the shape of matrices.

```python
[m*n].[n*x] = [m*x]
[m*x].[x*y] = [m*y]
```

We take the $[m*n]$ as the input matrix this is a $[4*3]$ matrix.

Similarly the output $y$ is a $[4*1]$ matrix; so we have $[m*y] =[4*1]$

So we have

```python
m=4
n=3
x=?
y=1
```

Lets then create our two weight matrices of the above shapes, that represent the two layers of the neural network.

```python
w0 = x
w1 = np.random.random((3,4))
w2 = np.random.random((4,1))
```

We can have an array of the weights to loop through, but for the time being let's hard-code these. Note that 'np' stands for the popular numpy array library in Python.

We also need to code in our non linearity.We will use the Sigmoid function here.

```python
def sigmoid(x):
    return 1/(1+np.exp(-x))

# derivative of the sigmoid
def derv_sigmoid(x):
   return x*(1-x)
```

With this we can have the output of first, second and third layer, using our equation of neural network forward propagation.

```python
a0 = x
a1 = sigmoid(np.dot(a0,w1))

a2 = sigmoid(np.dot(a1,w2))
```

a2 is the calculated output from randomly initialized weights. So lets calculate the error by subtracting this from the expected value and taking the MSE.

$$
 C = \frac{1}{2} \|y-a^l\|^2
$$

```python
c0 = ((y-a2)**2)/2
```

Now we need to use the back-propagation algorithm to calculate how each weight has influenced the error and reduce it proportionally.

---

We use this to update weights in all the layers and do forward pass again, re-calculate the error and loss, then re-calculate the error gradient $\frac{\partial C}{\partial w}$ and repeat

$$\begin{aligned}

w^2 = w^2 - (\frac {\partial C}{\partial w^2} )*learningRate \\ \\

w^1 = w^1 - (\frac {\partial C}{\partial w^1} )*learningRate

\end{aligned}$$

Let's update the weights as per the formula (3) and (5)

$$\begin{aligned}

\mathbf{
\frac {\partial C}{\partial w^2} = \sigma' (z^{2})*(a^2-y) \quad \rightarrow (3) } \\ \\

\mathbf{
\frac {\partial C}{\partial w^1} =\frac {\partial C}{\partial(a^2)} *w^2 . \sigma'(z^1)  =(a^2-y)*w^2 . \sigma'(z^1)\quad \rightarrow \mathbb (5)
}
\end{aligned}$$

```python
dc_dw2 =  (a2-y)*der_sigmoid(np.dot(a1,w2))
dc_dw1 =  (a2-y)*w2*der_sigmoid(np.dot(a0,w1))

```

Todo - Finish program

Reference  
- https://cedar.buffalo.edu/~srihari/CSE574/Chap5/Chap5.3-BackProp.pdf