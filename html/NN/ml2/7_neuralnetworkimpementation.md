# Neural Network Implementation


With the derivatives of the Cost function derived in Chapter 5 (MSE loss with Sigmoid activation), we can code the network. At the end of the chapter we switch the output layer to the Softmax and Cross Entropy Loss from Chapter 6.

We will use matrices to represent input and weight matrices.
 
> **Important Note on Matrix Conventions**
>
> In the previous chapters (Matrix Calculus), we derived equations assuming **Column Vectors** (where input $x$ is $N \times 1$). However, in standard Python/NumPy implementations (like the one below), we typically use **Batch Processing** where inputs are **Row Vectors** (where input $X$ is $BatchSize \times Features$).
>
> This means:
>
> - Input $X$ has shape `(BatchSize, InputFeatures)`.
> - Weight $W$ has shape `(InputFeatures, OutputFeatures)`.
> - Forward pass is $Z = X \cdot W$ (instead of $W \cdot x$).
> - This effectively transposes the standard mathematical notation. The code below follows this "Row Vector/Batch" convention.

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
Note you can change the output and try to train the Neural network

This is a 4*1 matrix that represents the expected output. That is for input [0,0,1] the output is [0], for [0,1,1] it is [1], for [1,0,1] it is [1] and for [1,1,1] it is [0].

**A neural network is implemented as a set of matrices representing the weights of the network.**

Let's create a two layered network. Before that please note the formula for the neural network

So basically the output at layer l is the dot product of the weight matrix of layer l and input of the previous layer.

Now let's see how the matrix dot product works based on the shape of matrices.

```text
[m*n].[n*h] = [m*h]
[m*h].[h*k] = [m*k]
```

We take the $[m*n]$ as the input matrix this is a $[4*3]$ matrix.

Similarly the output is a $[4*1]$ matrix; so we have $[m*k] = [4*1]$

So we have

```python
m=4   # rows: the 4 training examples
n=3   # input features
h=4   # hidden units (a free choice, we pick 4)
k=1   # output features
```

Lets then create our two weight matrices of the above shapes, that represent the two layers of the neural network.

```python
weight1 = np.random.random((n,h))   # (3,4): input -> hidden
weight2 = np.random.random((h,k))   # (4,1): hidden -> output
```

We can have an array of the weights to loop through, but for the time being let's hard-code these. Note that 'np' stands for the popular numpy array library in Python.

We also need to code in our non-linearity. We will use the Sigmoid function here.

```python
def sigmoid(x):
    return 1/(1+np.exp(-x))

# derivative of the sigmoid
def derv_sigmoid(x):
   return sigmoid(x)*(1-sigmoid(x))
```

With this we can have the output of first, second and third layer, using our equation of neural network forward propagation.

```python
a0 = x
a1 = sigmoid(np.dot(a0,weight1))

a2 = sigmoid(np.dot(a1,weight2))
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

$$
\begin{aligned}
w^2 = w^2 - (\frac {\partial C}{\partial w^2} )*learningRate \\ \\
w^1 = w^1 - (\frac {\partial C}{\partial w^1} )*learningRate
\end{aligned}
$$

Let's update the weights as per the formulas derived in Chapter 5 (Backpropagation with Matrix Calculus):

$$
\delta^2 = (a^2 - y) \odot \sigma'(z^2)
$$

$$
\frac {\partial C}{\partial w^2} = \delta^2 \, (a^1)^T \quad \rightarrow \text{Eq (3)}
$$

$$
\frac {\partial C}{\partial w^1} = \left( \left((w^2)^T \delta^2\right) \odot \sigma'(z^1) \right) (a^0)^T \quad \rightarrow \text{Eq (5)}
$$

## A Two layered Neural Network in Python

Below is a two layered Network; I have used the code from http://iamtrask.github.io/2015/07/12/basic-python-network/ as the basis. With minor changes to fit into how we derived the equations.

```python
import numpy as np
# seed random numbers to make calculation deterministic 
np.random.seed(1)

# pretty print numpy array
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# let us code our sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# let us add a method that takes the derivative of x as well
def derv_sigmoid(x):
   return sigmoid(x)*(1-sigmoid(x))

#---------------------------------------------------------------

# Two layered NW. Using from (1) and the equations we derived as explanations
# (1) http://iamtrask.github.io/2015/07/12/basic-python-network/
#---------------------------------------------------------------

# set learning rate as 1 for this toy example
learningRate = 1

# input x, also used as the training set here
x = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])

# desired output for each of the training set above
y = np.array([[0,1,1,0]]).T

# Explanation - as long as input has two ones, but not three, output is One
"""
Input [0,0,1]  Output = 0
Input [0,1,1]  Output = 1
Input [1,0,1]  Output = 1
Input [1,1,1]  Output = 0
"""

# Randomly initialized weights
weight1 =  np.random.random((3,4)) 
weight2 =  np.random.random((4,1)) 

# Activation to layer 0 is taken as input x
a0 = x

iterations = 1000
for iter in range(0,iterations):

  # Forward pass - Straight Forward
  z1= np.dot(x,weight1)
  a1 = sigmoid(z1) 
  z2= np.dot(a1,weight2)
  a2 = sigmoid(z2) 
  if iter == 0:
    print("Initial Output \n",a2)

  # Backward Pass - Backpropagation 
  dC_da2  = (a2-y)
  #---------------------------------------------------------------
  # Error term of the last layer: delta^2 = (a^2 - y) * sigmoid'(z^2)
  # Eq (3), row/batch form ---> dC_dw2 = a1.T . delta2
  #---------------------------------------------------------------

  delta2  = dC_da2*derv_sigmoid(z2)
  dC_dw2  = a1.T.dot(delta2)
  
  #---------------------------------------------------------------
  # Error term of the hidden layer: (delta2 . w2.T) * sigmoid'(z^1)
  # Eq (5), row/batch form ---> dC_dw1 = a0.T . ((delta2 . w2.T) * sigmoid'(z1))
  #---------------------------------------------------------------

  dC_dw1 =  np.dot(delta2, weight2.T) * derv_sigmoid(z1)
  dC_dw1 = a0.T.dot(dC_dw1)

  #---------------------------------------------------------------
  #Gradient descent
  #---------------------------------------------------------------
 
  weight2 = weight2 - learningRate*(dC_dw2)
  weight1 = weight1 - learningRate*(dC_dw1)


print("New output",a2)

#---------------------------------------------------------------
# Training is done, weight2 and weight2 are primed for output y
#---------------------------------------------------------------

# Lets test out, two ones in input and one zero, output should be One
x = np.array([[1,0,1]])
z1= np.dot(x,weight1)
a1 = sigmoid(z1) 
z2= np.dot(a1,weight2)
a2 = sigmoid(z2) 
print("Output after Training is \n",a2)
```

Output

```console
Initial Output 
 [[ 0.758]
 [ 0.771]
 [ 0.791]
 [ 0.801]]
New output [[ 0.028]
 [ 0.925]
 [ 0.925]
 [ 0.090]]
Output after Training is 
 [[ 0.925]]
 ```

We have trained the NW for getting the output similar to $y$; that is [0,1,1,0]

## The Same Network with Softmax and Cross Entropy

The network above uses the MSE loss with a Sigmoid output from Chapter 5. Let's now switch the output layer to the Softmax and Cross Entropy Loss we derived in Chapter 6. Three things change:

1. The target is **one-hot encoded**. With two classes, output 0 becomes $[1,0]$ and output 1 becomes $[0,1]$, so $y$ is now a $[4*2]$ matrix and the last weight matrix is $[4*2]$.
2. The output activation is **Softmax**, so each row of $a^2$ is a probability distribution over the two classes.
3. The error term at the output is simply (EqA1.1)

$$
\delta^2 = \frac{\partial C}{\partial z^2} = p - y
$$

There is no separate $\sigma'(z^2)$ factor; the derivative of Softmax cancels against the derivative of Cross Entropy. The hidden layer is unchanged and uses EqA2.1:

$$
\delta^1 = \left( (W^2)^T \delta^2 \right) \odot \sigma'(z^1)
$$

```python
import numpy as np
np.random.seed(1)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derv_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

# softmax over each row (each row is one example)
def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))  # subtract max for numerical stability
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(p, y):
    return -np.sum(y * np.log(p))

learningRate = 1

x = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])

# the same labels as before, now one-hot encoded over two classes
# class 0 -> [1,0], class 1 -> [0,1]
y = np.array([ [1,0],[0,1],[0,1],[1,0] ])

weight1 = np.random.random((3,4))   # input -> hidden
weight2 = np.random.random((4,2))   # hidden -> 2 output classes

a0 = x

iterations = 1000
for iter in range(0,iterations):

  # Forward pass
  z1 = np.dot(a0,weight1)
  a1 = sigmoid(z1)
  z2 = np.dot(a1,weight2)
  a2 = softmax(z2)            # a2 is P, the Softmax output
  if iter == 0:
    print("Initial Loss", round(cross_entropy(a2,y),3))

  # Backward pass
  #---------------------------------------------------------------
  # EqA1.1 ---> delta2 = dC_dz2 = p - y
  # No separate sigmoid'(z2) term: Softmax + Cross Entropy cancel it out
  #---------------------------------------------------------------
  delta2 = a2 - y
  dC_dw2 = a1.T.dot(delta2)                      # EqA1, row/batch form

  #---------------------------------------------------------------
  # EqA2.1 ---> delta1 = (delta2 . w2.T) * sigmoid'(z1)
  #---------------------------------------------------------------
  delta1 = np.dot(delta2, weight2.T) * derv_sigmoid(z1)
  dC_dw1 = a0.T.dot(delta1)                      # EqA2, row/batch form

  # Gradient descent
  weight2 = weight2 - learningRate*dC_dw2
  weight1 = weight1 - learningRate*dC_dw1

print("Final Loss", round(cross_entropy(a2,y),3))
print("Class probabilities after training \n", a2)
print("Predicted class", np.argmax(a2, axis=1))
```

Output

```console
Initial Loss 3.286
Final Loss 0.009
Class probabilities after training 
 [[ 0.999  0.001]
 [ 0.003  0.997]
 [ 0.001  0.999]
 [ 0.995  0.005]]
Predicted class [0 1 1 0]
```

The predicted classes are again [0,1,1,0]. Compare the two backward passes: only the line computing `delta2` is different. This is why Softmax with Cross Entropy is the standard output layer for classification; the gradient at the output is just the difference between the predicted probabilities and the truth.

## References

- [A Neural Network in 11 lines of Python - I Am Trask](http://iamtrask.github.io/2015/07/12/basic-python-network/)
- [Colab Notebook](https://colab.research.google.com/drive/1uB6N4qN_-0n8z8ppTSkUQU8-AgHiD5zD?usp=sharing)

[Index](index.md)
