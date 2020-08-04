# Walk through of a Simple Neural Network

Let's take the simple neural network in Python given here and walk through the same. 

I am using the example here http://iamtrask.github.io/2015/07/12/basic-python-network/ as the base of this article

## Step 1.

Let's write the  equation of the neural network

```
x is the Input.
y is the Output.
l is a layer of the Neural Network.
a is the activation function
```

$$
 x \rightarrow a^{l-1} \rightarrow  a^{l} \rightarrow  y
 $$
 
 and in the activation function we take the sigmoid of the input of the previous layer $a^{l-1}$. That is in layer 2, the input $x$ is the input

$$
  a^{l} = \sigma(w^l a^{l-1}+b^l).
$$

This is not so difficult to code up. Let's do this. I am following the blog and code here http://iamtrask.github.io/2015/07/12/basic-python-network/ adding little more explanation for each of the steps, from what we have learned.

We will use matrices to code up our network

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

A neural network can be implemented as a set of arrays representing the weights of the network.

Let's create a 2 layered network. Before that please not the formula for the neural network

$$
  z^{l} =(w^l a^{l-1}+b^l).
$$

If we create a dummy input $a^0 =1$, then we can shift the basis in the above equation to $w^0$. 

$$
  \sum _{l=1}^{n}z^{l} =(w^l a^{l-1}).
$$
Ignore basis for the time being.

So basically the output at layer l is the dot product of the weight matrix of layer l and input of the previous layer.

Now let's see how the matrix dot product works based on the shape of matrices.

```
[m*n].[n*x] = [m*x]
[m*x].[x*y] = [m*y]
```

We take the $[m*n]$ as the input matrix this is a $[4*3]$ matrix.

Similarly the output $y$ is a $[4*1]$ matrix; so we have $[m*y] =[4*1]$

So we have

```
m=4
n=3
x=?
y=1
```
Lets then create our two weight matrices of the above shapes, that represent the two layers of the neural network.

```
w0 = x
w1 = np.random.random((3,4)) 
w2 = np.random.random((4,1)) 
```
We can have an array of the weights to loop through, but for the time being let's hard-code these. Note that 'np' stands for the popular numpy array library in Python.

We also need to code in our non linearity.We will use the Sigmoid function here.

```
def sigmoid(x):
    return 1/(1+np.exp(-x))

# derivative of the sigmoid
def derv_sigmoid(x):
   return x*(1-x)
```

With this we can have the output of first, second and third layer, using our equation of neural network forward propagation.

(Note - ignore the basis term for now)

$$
  a^{l} = \sigma(w^l a^{l-1}+b^l).
$$

```
a0 = x
```
$$
  a^{1} = \sigma(w^1 a^{0}+b^l).
$$

```
a1 = sigmoid(np.dot(a0,w1))
```
$$
  a^{2} = \sigma(w^2 a^{1}+b^l).
$$
```
a2 = sigmoid(np.dot(a1,w2))
```

a2 is the calculated output from randomly initialized weights. So lets calculate the error by subtracting this from the expected value and taking the MSE.

$$
 C = \frac{1}{2} \|y-a^l\|^2
$$

```
c0 = ((y-a2)**2)/2
```
Now we need to use the back-propagation algorithm to calculate how each weight has influenced the error and reduce it proportionally.

$$
\delta C_0/\delta w^l = \delta z^l/\delta w^l . \delta a^l/\delta z^l . \delta C_0/\delta a^l
$$
$$
z^l = a^{l-1}.w^l
$$

c0w2 = 

$$

\delta C_0/\delta w^2 = \delta z^2/\delta w^2 . \delta a^2/\delta z^2 . \delta C_0/\delta a^2

$$

```
a2 = sigmoid(np.dot(a1,w2))
z2 = np.dot(a1,w2)
```

TODO