# Chapter 5: Back Propagation for a Two layered Neural Network

Let's take the simple neural network and walk through the same, first going through the maths and then the implementation.

Let's write the  equation of the following two layer neural network

```
x is the Input 
y is the Output.
l is the number of layers of the Neural Network.
a is the activation function ,(we use sigmoid here)
```

$$
 x \rightarrow a^{l-1} \rightarrow  a^{l} \rightarrow  y
$$

Repeating here the previous equation's that we derived in the previous chapter 

$$ \mathbf{
\frac {\partial C}{\partial w^2} =  a^1* \sigma' (z^{2})*(a^2-y) \quad \rightarrow (A) }
$$

$$
\mathbf{
\frac {\partial C}{\partial w^1} =a^0* \sigma'(z^1)*(a^2-y).\sigma'(z^2).w^2 \quad \rightarrow \mathbb (B)
}
$$

---

Note that weight is a Vector and we need to use the Vector product /dot product where weights are concerned. We will do an implementation to test out these equations to be sure.


With equations (A) and (B) we can calculate the gradient of the Loss function with respect to weights in any layel - in this example $\frac {\partial C}{\partial w^1},\frac {\partial C}{\partial w^2}$

We now need to adjust the previous weight, by gradient descent.

So using the above gradients we get the new weights iteratively like below. If you notice this is exactly what is happening in gradient descent as well; only chain rule is used to calculate the gradients here. Backpropagation is the algorithm that helps calculate the gradients for each layer.

---

$$

  W^{l-1}_{new} = W^{l-1}_{old} - learningRate* \delta C_0/ \delta w^{l-1}

$$

---

## Implementation

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