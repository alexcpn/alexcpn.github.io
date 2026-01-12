# The Mathematical Intuition Behind Deep Learning

Alex Punnen \
&copy; All Rights Reserved

---

# Chapter 7

## Back Propagation in Full - With Softmax & CrossEntropy Loss


Let's think of a $l$ layered neural network whose input is $x=a^0$ and output is $a^l$.In this network we will be using the **sigmoid ($\sigma$ )** function as the activation function for all layers except the last layer $l$. For the last layer we use the **Softmax activation function**. We will use the **Cross Entropy Loss** as the loss function.

This is how a proper Neural Network should be. 

 
## The Neural Network Model

I am writing this out, without index notation, and with the super script representing just the layers of the network.

$$
\mathbf {
\begin{aligned}
 a^0 \rightarrow
      \underbrace{\text{hidden layers}}_{a^{l-2}} 
      \,\rightarrow
      \underbrace{W^{l-1} a^{l-2}+b^{l-1}}_{z^{l-1} }
      \,\rightarrow
      \underbrace{\sigma(z^{l-1})}_{a^{l-1}}
    \,\rightarrow
     \underbrace{W^l a^{l-1}+b^l}_{z^{l}/logits }
    \,\rightarrow
    \underbrace{P(z^l)}_{\vec P/ \text{softmax} /a^{l}}
    \,\rightarrow
    \underbrace{L ( \vec P, \vec Y)}_{\text{CrossEntropyLoss}}
\end{aligned}
}
$$

 $Y$ is the target vector or the Truth vector. This is a one hot encoded vector, example  $Y=[0,1,0]$, here the second element is the desired class.The training is done so that the CrossEntropyLoss is minimized using Gradient Descent algorithm.

$P$ is the Softmax output and is the activation of the last layer $a^l$. This is a vector. All elements of the Softmax output add to 1; hence this is a probability distribution unlike a Sigmoid output.The Cross Entropy Loss $L$ is a Scalar.

Note the Index notation is the representation an element of a Vector or a Tensor, and is easier to deal with while deriving out the equations.

**Softmax** (in Index notation)

Below I am skipping the superscript part, which I used to represent the layers of the network.

$$
\begin{aligned}
p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}
\end{aligned}
$$

This represent one element of the softmax vector, example $\vec P= [p_1,p_2,p_3]$

**Cross Entropy Loss** (in Index notation)

Here $y_i$ is the indexed notation of an element in the target vector  $Y$.

$$
\begin{aligned}
L = -\sum_j y_j \log p_j
\end{aligned}
$$

---


There are too many articles related to Back propagation, many of which are very good.However many explain in terms of index notation and though it is illuminating, to really use this with code, you need to understand how it translates to Matrix notation via Matrix Calculus and with help form StackOverflow related sites.

### CrossEntropy Loss with respect to Weight in last layer

$$
\mathbf {
\frac {\partial L}{\partial W^l} 
=  \color{red}{\frac {\partial L}{\partial z^l}} \cdot \color{green}{\frac {\partial z^l}{\partial W^l}} \rightarrow \quad EqA1
}
$$

Where
$$
L = -\sum_k y_k \log {\color{red}{p_k}} \quad \text{and} \quad p_j = \frac {e^{\color{red}{z_j}}} {\sum_k e^{z_k}}
$$

If you are confused with the indexes, just take a short example and substitute. Basically i,j,k etc are dummy indices used to illustrate in index notation the vectors.

I am going to drop the superscript $l$ denoting the layer number henceforth and focus on the index notation for the softmax vector $P$ and target vector $Y$

From [Derivative of Softmax Activation -Alijah Ahmed]

$$ \color{red}
{
\begin{aligned}

 \frac {\partial L}{\partial z_i} = \frac {\partial ({-\sum_k y_k \log {p_k})}}{\partial z_i}
   \\ \\ \text {taking the summation outside} \\ \\
   = -\sum_k y_k\frac {\partial ({ \log {p_k})}}{\partial z_i}
  \\ \\ \color{grey}{\text {since }
  \frac{d}{dx} (f(g(x))) = f'(g(x))g'(x) }
  \\ \\
  = -\sum_k y_k \cdot \frac {1}{p_k} \cdot \frac {\partial { p_k}}{\partial z_i}
\end{aligned}
}
$$

The last term $\frac {\partial { p_k}}{\partial z_i}$ is the derivative  of Softmax with respect to its inputs also called logits. This is easy to derive and there are many sites that describe it. Example [Derivative of SoftMax Antoni Parellada]. The more rigorous derivative via the Jacobian matrix is here [The Softmax function and its derivative-Eli Bendersky]

$$
 \color{red}
  {
  \begin{aligned}
   \frac {\partial { p_i}}{\partial z_i} = p_i(\delta_{ij} -p_j)
   \\ \\
   \delta_{ij} = 1 \text{ when i =j}
   \\
   \delta_{ij} = 0 \text{ when i} \ne \text{j}
  \end{aligned}
  }
$$

Using this above and from [Derivative of Softmax Activation -Alijah Ahmed]

$$ \color{red}
  {
  \begin{aligned}

 \frac {\partial L}{\partial z_i} = -\sum_k y_k \cdot \frac {1}{p_k} \cdot \frac {\partial { p_k}}{\partial z_i}
 \\ \\
  =-\sum_k y_k \cdot \frac {1}{p_k} \cdot p_i(\delta_{ij} -p_j) 
 \\ \\ \text{these i and j are dummy indices and we can rewrite  this as} 
\\ \\
=-\sum_k y_k \cdot \frac {1}{p_k} \cdot p_k(\delta_{ik} -p_i) 
\\ \\ \text{taking the two cases and adding in above equation } \\ \\
 \delta_{ik} = 1 \text{ when i =k} \text{ and } 
   \delta_{ik} = 0 \text{ when i} \ne \text{k}
   \\ \\
   = [- y_i \cdot \frac {1}{p_i} \cdot p_i(1 -p_i)]+[-\sum_{k \ne i}  y_k \cdot \frac {1}{p_k} \cdot p_k(0 -p_i) ]
    \\ \\
     = [- y_i \cdot \frac {1}{p_i} \cdot p_i(1 -p_i)]+[-\sum_{k \ne i}  y_k \cdot \frac {1}{p_k} \cdot p_k(0 -p_i) ]
  \\ \\
     = [- y_i(1 -p_i)]+[-\sum_{k \ne i}  y_k \cdot (0 -p_i) ]
      \\ \\
     = -y_i + y_i.p_i + \sum_{k \ne i}  y_k.p_i 
     \\ \\
     = -y_i + p_i( y_i + \sum_{k \ne i}  y_k) 
     \\ \\
     = -y_i + p_i( \sum_{k}  y_k) 
     \\ \\
     \text {note that } \sum_{k}  y_k = 1  \, \text{as it is a One hot encoded Vector}
     \\ \\
     = p_i - y_i
     \\ \\
     \frac {\partial L}{\partial z^l}  = p_i - y_i \rightarrow \quad \text{EqA1.1}
\end{aligned}
}
$$

We need to put this back in $EqA1$. We now need to calculate the second term, to complete the equation

$$
\begin{aligned}
\frac {\partial L}{\partial W^l} 
=  \color{red}{\frac {\partial L}{\partial z^l}} \cdot \color{green}{\frac {\partial z^l}{\partial W^l}}
\\ \\
z^{l} = (W^l a^{l-1}+b^l) 
\\
 \color{green}{\frac {\partial z^l}{\partial W^l} = (a^{l-1})^T}
 \\ \\ \text{Putting all together} \\ \\
 \frac {\partial L}{\partial W^l} = (p - y) \cdot (a^{l-1})^T \quad  \rightarrow \quad \mathbf  {EqA1}
\end{aligned}
$$

## Gradient descent

Using Gradient descent we can keep adjusting the last layer like

$$
     W{^l} = W{^l} -\alpha \cdot \frac {\partial L}{\partial W^l} 
$$

Now let's do the derivation for the inner layers

## Derivative of Loss with respect to Weight in Inner Layers



The trick here is to find the derivative of the Loss with respect to the inner layer as a composition of the partial derivative we computed earlier. And also to compose each partial derivative as partial derivative with respect to either $z^x$ or $w^x$ but not with respect to $a^x$. This is to make derivatives easier and intuitive to compute.


$$
\begin{aligned}
\frac {\partial L}{\partial W^{l-1}} 
=  \color{blue}{\frac {\partial L}{\partial z^{l-1}}} \cdot
     \color{green}{\frac {\partial z^{l-1}}{\partial W^{l-1}}} \rightarrow \text{EqA2}
\end{aligned}
$$

We  represent the first  part  in terms of what we computed earlier ie $\color{blue}{\frac {\partial L}{\partial z^{l}}}$


$$
\begin{aligned}
\color{blue}{\frac {\partial L}{\partial z^{l-1}}} =
\color{blue}{\frac {\partial L}{\partial z^{l}}}.
    \frac {\partial z^{l}}{\partial a^{l-1}}.
    \frac {\partial a^{l-1}}{\partial z^{l-1}} \rightarrow \text{ Eq with respect to Prev Layer}
  \\ \\
  \color{blue}{\frac {\partial L}{\partial z^{l}}} = \color{blue}{(p_i- y_i)}
  \text{ from the previous layer (from EqA1.1) } 
  \\ \\
   z^l = w^l a^{l-1}+b^l
    \text{ which makes }
    {\frac {\partial z^{l} }{\partial a^{l-1}} = w^l} \\
    \text{ and }
 a^{l-1} = \sigma (z^{l-1})     \text{ which makes }
\frac {\partial a^{l-1}}{\partial z^{l-1}} = \sigma \color{red}{'} (z^{l-1} )


\\ \\ \text{ Putting together we get the first part of Eq A2 }
\\\\
\color{blue}{\frac {\partial L}{\partial z^{l-1}}} = \left( (W^l)^T \cdot \color{blue}{(p- y)} \right) \odot \sigma \color{red}{'} (z^{l-1} ) \rightarrow \text{EqA2.1 }
\\ \\
 z^{l-1} = W^{l-1} a^{l-2}+b^{l-1}
    \text{ which makes }
    \color{green}{\frac {\partial z^{l-1}}{\partial W^{l-1}}=(a^{l-2})^T}
\\ \\
\frac {\partial L}{\partial W^{l-1}} 
=  \color{blue}{\frac {\partial L}{\partial z^{l-1}}} \cdot
     \color{green}{\frac {\partial z^{l-1}}{\partial W^{l-1}}} = \left( \left( (W^l)^T \cdot \color{blue}{(p- y)} \right) \odot \sigma '(z^{l-1} ) \right) \cdot \color{green}{(a^{l-2})^T}
\end{aligned}
$$

**Note** All the other layers should use the previously calculated value of  $\color{blue}{\frac {\partial L}{\partial z^{l-i}}}$ where  $i= current layer-1$

$$
\begin{aligned}
\frac {\partial L}{\partial W^{l-2}} 
=  \color{blue}{\frac {\partial L}{\partial z^{l-2}}} \cdot
     \color{green}{\frac {\partial z^{l-2}}{\partial W^{l-2}}}
\end{aligned}
$$



## Implementation in Python

Here is an implementation of a relatively simple Convolutional Neural Network to test out the forward and back-propagation algorithms given above [https://github.com/alexcpn/cnn_in_python](https://github.com/alexcpn/cnn_in_python). The code is well commented and you will be able to follow the forward and backward propagation with the equations above. Note that the full learning cycle is not completed; but rather a few Convolutional layers, forward propagation and backward propogation for last few layers.

## Gradient descent

Using Gradient descent we can keep adjusting the inner layers like

$$
     W{^{l-1}} = W{^{l-1}} -\alpha \cdot \frac {\partial L}{\partial W^{l-1}} 
$$


## Some Implementation details

Feel free to skip this section. These are some doubts that can come during implementation,and can be referred to if needed.

**From Index Notation to Matrix Notation**

The above equations are correct only as far as the index notation is concerned. But practically we work with Weight matrices, and for that we need to write this Equation in *Matrix Notation*. For that some of the terms becomes Transposes, some matrix multiplication (dot product style) and some Hadamard product. ($\odot$). This is illustrated and commented in the code and deviates from the equations as is,

Example 
$$
\frac{\partial z^2}{\partial W^2} = (a^{1})^T 
$$

### The Jacobian Matrix (and why Hadamard appears)

Let $x \in \mathbb{R}^n$ be an input vector and let an elementwise function $f$ (e.g., sigmoid) produce an output vector $a \in \mathbb{R}^n$:

$$
a = f(x), \quad a_i = f(x_i)
$$

In the scalar case, the derivative is just:

$$
\frac{d}{dx} f(x) = f'(x)
$$

In the vector case, the derivative of a vector-valued function with respect to a vector input is the Jacobian matrix:

$$
J = \frac{\partial a}{\partial x} \in \mathbb{R}^{n \times n}, \quad J_{ij} = \frac{\partial a_i}{\partial x_j}
$$

For an elementwise function $a_i = f(x_i)$, each output component depends only on the corresponding input component, so:

$$
\frac{\partial a_i}{\partial x_j} = 
\begin{cases} 
f'(x_i) & \text{if } i = j \\
0 & \text{if } i \neq j 
\end{cases}
$$

and therefore the Jacobian is diagonal:

$$
\frac{\partial a}{\partial x} = \text{diag}(f'(x))
$$

**Why this becomes a Hadamard product in backprop**

Suppose we have a vector $v$ coming from later in the chain rule (e.g., $v = \frac{\partial C}{\partial a}$). Then:

$$
\frac{\partial C}{\partial x} = \left( \frac{\partial a}{\partial x} \right)^T \frac{\partial C}{\partial a} = \text{diag}(f'(x)) \cdot v = f'(x) \odot v
$$

So the Hadamard product appears because the Jacobian of an elementwise function is diagonal, and multiplying by a diagonal matrix is the same as elementwise multiplication.

 [ref 1]:https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf
 [ref 2]:https://aew61.github.io/blog/artificial_neural_networks/1_background/1.b_activation_functions_and_derivatives.html


----

What is basically done is to flatten the Matrix out

$$
\begin{aligned}
\text{Let's take a 2x2 matrix , X } =
\begin{bmatrix}
                x_{ {1}{1} }  & x_{ {1}{2} } \\
                x_{ {2}{1} }  & x_{ {2}{2} }
\end{bmatrix}
\end{aligned}
$$
On which an element wise operation is done  $a_{ {i}{j} } = \sigma ({x_{ {i}{j} }})$
Writing that out as matrix $A$
$$
\begin{aligned}
A =
\begin{bmatrix}
                a_{ {1}{1} }  & a_{ {1}{2} }   \\
                a_{ {2}{1} }  & a_{ {2}{2} }   \\
\end{bmatrix}
\end{aligned}
$$

The partial derivative of the elements of A with its inputs is $\frac {\partial A }{\partial x_{ {i}{j} }}$

$$
\begin{aligned}
\frac {\partial \vec A }{\partial X} =
\begin{bmatrix}
                a_{ {1}{1} }  & a_{ {1}{2} }  &  a_{ {2}{1} }  & a_{ {2}{2} }   \\
\end{bmatrix}
\end{aligned}
$$
We vectorized the matrix; Now we need to take the partial derivative of the vector with each element of the matrix $X$

$$
\begin{aligned}
\frac {\partial \vec A }{\partial X} =
\begin{bmatrix}
\frac{\partial  a_{ {1}{1} } }{\partial x_{ {1}{1} }} &   \frac{\partial  a_{ {1}{2} } }{\partial x_{ {1}{1} }} &   \frac{\partial  a_{ {2}{1} } }{\partial x_{ {1}{1} }} &   \frac{\partial  a_{ {2}{2}} }{\partial x_{ {1}{1}}}  \\ \\
 \frac{\partial  a_{ {1}{1}} }{\partial x_{ {1}{2}}} &   \frac{\partial  a_{ {1}{2}} }{\partial x_{ {1}{2}}} &   \frac{\partial  a_{ {2}{1}} }{\partial x_{ {1}{2}}} &   \frac{\partial  a_{ {2}{2}} }{\partial x_{ {1}{2}}}  \\ \\
\frac{\partial  a_{ {1}{1}} }{\partial x_{ {2}{1}}} &   \frac{\partial  a_{ {1}{2}} }{\partial x_{ {2}{1}}} &   \frac{\partial  a_{ {2}{1}} }{\partial x_{ {2}{1}}} &   \frac{\partial  a_{ {2}{2}} }{\partial x_{ {2}{1}}}  \\ \\
\frac{\partial  a_{ {1}{1}} }{\partial x_{ {2}{2}}} &   \frac{\partial  a_{ {1}{2}} }{\partial x_{ {2}{2}}} &   \frac{\partial  a_{ {2}{1}} }{\partial x_{ {2}{2}}} &   \frac{\partial  a_{ {2}{2}} }{\partial x_{ {2}{2}}}  \\ \\
\end{bmatrix}
\end{aligned}
$$
The non diagonal terms are of the form  $\frac{\partial  a_{ {i}{j}} }{\partial x_{ {k}{k}}}$ and reduce to 0 and we get the resultant Jacobian Matrix as


$$
\begin{aligned}
\frac {\partial \vec A }{\partial X} =
\begin{bmatrix}
\frac{\partial  a_{ {i}{j}} }{\partial x_{ {i}{i}}} & \cdot \cdot \cdot & 0 \\
 0 & \frac{\partial  a_{ {i}{j}} }{\partial x_{ {i}{i}}} & \cdot \cdot \cdot  \\
 \cdot \cdot \cdot  \\
 \cdot \cdot \cdot  & \cdot \cdot \cdot & \frac{\partial  a_{ {N}{N}} }{\partial x_{ {N}{N}}} 
\end{bmatrix}
\end{aligned}
$$



Hence  $\frac{\partial a_{ {i}{j}}}{\partial X}$ can be written as $\text{ diag}(f'(X))$ ; $(A =f(X))$


Note that Multiplication of a vector by a diagonal matrix is element-wise multiplication or the Hadamard product; *And matrices in Deep Learning implementation can be seen as stacked vectors for simplification.*

More details about this here [Jacobian Matrix for Element wise Operation on a Matrix (not Vector)](https://math.stackexchange.com/questions/4397390/jacobian-matrix-of-an-element-wise-operation-on-a-matrix)

Note that another way of interpreting this as treating weights as Tensor and then certain Jacobian operation can be treated as between Tensors and Vectors. 



## References
 
 Easier to follow (without explicit Matrix Calculus) though not really correct
 - [Supervised Deep Learning Marc'Aurelio Ranzato DeepMind]  
Easy to follow but lacking in some aspects
- [Notes on Backpropagation-Peter Sadowski]
Slightly hard to follow using the Jacobian 
 - [The Softmax function and its derivative-Eli Bendersky]
More difficult to follow with proper index notations (I could not) and probably correct
 - [Backpropagation In Convolutional Neural Networks Jefkine]


  [A Primer on Index Notation John Crimaldi]: https://web.iitd.ac.in/~pmvs/courses/mcl702/notation.pdf
  
  [The Matrix Calculus You Need For Deep Learning Terence,Jermy]:https://arxiv.org/pdf/1802.01528.pdf

  [The Matrix Calculus You Need For Deep Learning (Derivative with respect to Bias) Terence,Jermy]: https://explained.ai/matrix-calculus/#sec6.2
  
  [Neural Networks and Deep Learning Michel Neilsen]: http://neuralnetworksanddeeplearning.com/chap2.html


  [Supervised Deep Learning Marc'Aurelio Ranzato DeepMind]: https://bfeba431-a-62cb3a1a-s-sites.googlegroups.com/site/deeplearningcvpr2014/ranzato_cvpr2014_DLtutorial.pdf?attachauth=ANoY7cqPhkgQyNhJ9E7rmSk-RTdMYSYqpfJU2gPlb9cWH_4a1MbiYPq_0ihyuolPiYDkImyr9PmA-QwSuS8F3OMChiF97XTDD_luJqam70GvAC4X6G6KlU2r7Pv1rqkHaMbmXpdtXJHAveR_jWf1my_IojxFact87u2-1YXtfJIwYkhBwhMsYagICk-P6X9ktA0Pyjd601tboSlX_UGftX1vB57-tS6bdAkukhmSRLU-ZiF4RdJ_sI3YAGaaPYj1KLWFpkFa_-XG&attredirects=1
  
  [lecun-ranzato]: https://cs.nyu.edu/~yann/talks/lecun-ranzato-icml2013.pdf
  
  [Notes on Backpropagation-Peter Sadowski]: https://www.ics.uci.edu/~pjsadows/notes.pdf
  
  [The Softmax function and its derivative-Eli Bendersky]: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

  [Python Implementation of Jacobian of Softmax with respect to Logits Aerin Kim]: https://stackoverflow.com/a/46028029/429476
  
  [Derivative of Softmax Activation -Alijah Ahmed]: https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
  
  [Dertivative of SoftMax Antoni Parellada]: https://stats.stackexchange.com/a/267988/191675

  [Backpropagation In Convolutional Neural Networks Jefkine]: https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/]

  [Finding the Cost Function of Neural Networks Chi-Feng Wang]: https://towardsdatascience.com/step-by-step-the-math-behind-neural-networks-490dc1f3cfd9

  [Vector Derivatives]: http://cs231n.stanford.edu/vecDerivs.pdf
 
