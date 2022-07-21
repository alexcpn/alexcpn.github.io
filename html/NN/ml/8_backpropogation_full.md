# The Maths behind Neural Networks

Alex Punnen \
&copy; All Rights Reserved \

---

[Contents](index.md)

# Chapter 8

## Back Propagation in Full - With Softmax & CrossEntropy Loss

The previous chapters should have given an intuition of Back Propagation. Let's now dive much deeper into Back Propogation. We will need all the information covered in the previous chapters, plus a bit more involved mathematical concepts.

It is good to remember here Geoffrey Hinton's talk available in Youtube - All this was invented not out of some mathematical model; but based on trial and error and checking what works. So do not treat this as distilled science. This is ever evolving.

Let's think of a $l$ layered neural network whose input is $x=a^0$ and output is $a^l$.In this network we will be using the **sigmoid ($\sigma$ )** function as the activation function for all layers except the last layer $l$. For the last layer we use the **Softmax activation function**. We will use the **Cross Entropy Loss** as the loss function.

---

Below are some of the concepts that we had already covered in brief in previous chapters; and some which we have not touched previously; but without which it will not be possible to build a practical deep neural network solution.

- Understand what **Scalar's, Vectors, Tensors** are and that Vectors and Tensors are written as matrices and Vector is one dimension matrix whereas Tensor's are many dimensional usually. (Technically a Vector is also a Tensor). After this, you can forget about Tensors and think only of Vectors Matrices and Scalar. Mostly just matrices.

- That **linear algebra for matrices** that will be used is just properties of matrix multiplication and addition that you already know. A linear equation of the form $$y= m*x +c$$ in matrix form used in a neural network is $z_l = w_l* a_{l-1} + b_l$.

- The **Index notation for dealing with Vectors and Matrices** - [A Primer on Index Notation John Crimaldi]

- **Matrix multiplication** plays a major part and there are some parts that may be confusing
  
  - Example- **Dot Product** is defined only between Vectors, though many articles and tutorials will be using the dot product. Since each row of a multidimensional matrix acts like a Vector, the Numpy dot function(numpy.dot) works for matrix multiplication for non-vectors as well. Technically **numpy matmul** is the right one to use for matrix multiplication. $np.dot(A,B)$ is same as $np.matmul(A,B)$.numpy.**Numpy einsum** is also used for dimensions more than two. If A and B are two dimensional matrices $np.dot(A,B) = np.einsum('ij,jk->ik', A, B)$. And einsum is much easier than numpy.tensordot to work with. For Hadamard product **numpy.multiply**

    There is no accepted definition of matrix multiplication of dimensions higher than two!
  
  - **Hadamard product**. It is a special case of the element-wise multiplication of matrices of the same dimension. It is used in the magic of converting index notation to Matrix notation. You can survive without it, but you cannot convert to Matrix notation without understanding how. It is referred to in Michel Neilsen's famous book [Neural Networks and Deep Learning Michel Neilsen] in writing out the Error of a layer with respect to previous layers.

- Calculus, the concept of Derivatives, **Partial Derivatives, Gradient, Matrix Calculus, Jacobian Matrix**

  - That derivative of a function -the *derivative function* $f'(x)$, gives the slope or gradient of the function at any 'point'. As it is the rate of change of one variable with respect to to another. Visually, say for a function in 2D space , say a function representing a line segment, that means change in Y for a change in X - rise over run,slope.
  
  - For multi variable function, example a Vector function, we need the rate of change of many variables with respect to to another, we do so via `Partial derivatives`  concept - notation $\partial$ ; and the gradient becomes a Vector of partial derivatives. To visualize this, picture a hill, or a function of x,y,z variables that can be plotted in a 3D space, a ball dropped on this hill or graph goes down this `gradient vector` .To get the *derivative function* $f'(x,y,z)$ to calculate this gradient you need `multivariable calculus`, again something that you can ignore most of the time,except the slightly different rules while calculating the derivative function.

  - Take this a notch further and we reach the Jacobian matrix. For a Vector of/containing multivariable functions, the partial derivatives with respect to to say a Matrix or Vector of another function, gives a *Matrix of Partial Derivatives* called the `Jacobian Matrix`. And this is also a gradient matrix. It shows the 'slope' of the *derivative function* at a matrix of points. In our case the derivative of the Loss function (which is a scalar function) with respect to Weights (matrix), can be calculated only via intermediate terms, that include the derivative of the Softmax output (Vector) with respect to inputs (matrix) which is the Jacobian matrices. And that is matrix calculus. Again something that you can now ignore henceforth.

  - Knowing what a Jacobian is, and how it is calculated, you can blindly ignore it henceforth. The reason is that, most of the terms of the Jacobian evaluate to Zero for Deep learning application, and usually only the diagonal elements hold up, something which can be represented by index notation. *"So it's entirely possible to compute the derivative of the softmax layer without actual Jacobian matrix multiplication ...the Jacobian of the fully-connected layer is sparse.- [The Softmax function and its derivative-Eli Bendersky]"* 

    - Note -When you convert from Index notation to actual matrix notation, for example for implementation then you will need to understand how the index multiplication transforms to Matrix multiplication - transpose. Example from [The Matrix Calculus You Need For Deep Learning (Derivative with respect to Bias) Terence,Jermy]

$$
\frac{\partial z^2}{\partial w^2} = (1^{\rightarrow})^T* diag(a^1) =(a^{1})^T \quad
$$
 
- Calculus - **Chain Rule - Single variable, Multi variable Chain rule, Vector Chain Rule**
 
  - Chain rule is used heavily to break down the partial derivate of Loss function with respect to weight into a chain of easily differentiable intermediate terms

  - The Chain rule that is used is actually Vector Chain Rule , but due to nature of Jacobian matrices generated- sparse matrices, this reduces to resemble Chain rule of single variable or Multi-variable Chain Rule. Again the definite article to follow is [The Matrix Calculus You Need For Deep Learning (Derivative with respect to Bias) Terence,Jermy], as some authors refer as Multi variable Chain rule in their articles

    Single Variable Chain Rule
    $$
    \begin{aligned}
    y = f(g(x)) = f(u) \text{ where } u = g(x)
    \\ \\
    \frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}
    \end{aligned}
    $$

    **Vector Chain Rule**

     In the notation below, **y** is a Vector output and x is a scalar. Vectors are represented in bold letters though I have skipped it here.

    $$
    \begin{aligned}
    y = f(g(x))
    \\ \\
    \frac{ \partial y}{ \partial x} = \frac{ \partial y}{\partial g}*\frac{ \partial g}{\partial x}
    \end{aligned}
    $$

     Here $\frac{ \partial y}{\partial g}$ and $\frac{ \partial g}{\partial x}$ are two Jacobian matrices containing the set of partial derivatives. But since only the diagonals remain in deep learning application we can skip calculating the Jacobian and write in index notation as

     $$
    \begin{aligned}
    \frac{ \partial y}{ \partial x} = \frac{ \partial y_i}{\partial g_i} \frac{ \partial g_i}{\partial x_i}
    \end{aligned}
    $$

## The Neural Network Model

I am writing this out, without index notation, and with the super script representing just the layers of the network.

$$
\mathbf {
\bbox[10px, border:2px solid red] { \color{red}{
\begin{aligned}
 a^0 \rightarrow
    \bbox[5px, border:2px solid black]  {
      \underbrace{\text{hidden layers}}_{a^{l-2}} }
      \,\rightarrow
    \bbox[5px, border:2px solid black]  {  
      \underbrace{w^{l-1} a^{l-2}+b^{l-1}}_{z^{l-1} }
    }
      \,\rightarrow
    \bbox[5px, border:2px solid black]  {  
      \underbrace{\sigma(z^{l-1})}_{a^{l-1}}
    }
    \,\rightarrow
    \bbox[5px, border:2px solid black]  {  
     \underbrace{w^l a^{l-1}+b^l}_{z^{l}/logits }
    }
    \,\rightarrow
    \bbox[5px, border:2px solid black]  {  
    \underbrace{P(z^l)}_{\vec P/ \text{softmax} /a^{l}}
    }
    \,\rightarrow
  \bbox[5px, border:2px solid black]  {  
    \underbrace{L ( \vec P, \vec Y)}_{\text{CrossEntropyLoss}}
  }
\end{aligned}
}}}
$$

 $Y$ is the target vector or the Truth vector. This is a one hot encoded vector, example  $Y=[0,1,0]$, here the second element is the desired class.The training is done so that the CrossEntropyLoss is minimized using Gradient Loss algorithm.

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
## On to the rest of the explanation

There are too many articles related to Back propagation, many of which are very good.However many explain in terms of index notation and though it is illuminating, to really use this with code, you need to understand how it translates to Matrix notation via Matrix Calculus and with help form StackOverflow related sites.

### CrossEntropy Loss with respect to Weight in last layer

$$
\mathbf {
\frac {\partial L}{\partial w^l} 
=  \color{red}{\frac {\partial L}{\partial z^l}}.\color{green}{\frac {\partial z^l}{\partial w^l}} \rightarrow \quad EqA1
}
$$

Where
$$
\mathbf {
L = -\sum_k y_k \log \color{red}{p_k} \,\,and \,p_j = \frac {e^ \color{red}{z_j}} {\sum_k e^{z_k}}
}
$$

If you are confused with the indexes, just take a short example and substitute. Basically i,j,k etc are dummy indices used to illustrate in index notation the vectors.

I am going to drop the superscript $l$ denoting the layer number henceforth and focus on the index notation for the softmax vector $P$ and target vector $Y$

Repeating from [Derivative of Softmax Activation -Alijah Ahmed]

$$ \color{red}
  {
  \begin{aligned}

    \frac {\partial L}{\partial z_i} = \frac {\partial ({-\sum_j y_k \log {p_k})}}{\partial z_i}
   \\ \\ \text {taking the summation outside} \\ \\
   = -\sum_j y_k\frac {\partial ({ \log {p_k})}}{\partial z_i}
  \\ \\ \color{grey}{\text {since }
  \frac{d}{dx} (f(g(x))) = f'(g(x))g'(x) }
  \\ \\
  = -\sum_k y_k * \frac {1}{p_k} *\frac {\partial { p_k}}{\partial z_i}
  
\end{aligned}
}
$$

The last term $\frac {\partial { p_k}}{\partial z_i}$ is the derivative  of Softmax with respect to it's inputs also called logits. This is easy to derive and there are many sites that describe it. Example [Derivative of SoftMax Antoni Parellada]. The more rigorous derivative via the Jacobian matrix is here [The Softmax function and its derivative-Eli Bendersky]

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

Using this above and repeating from [Derivative of Softmax Activation -Alijah Ahmed]

$$ \color{red}
  {
  \begin{aligned}

 \frac {\partial L}{\partial z_i} = -\sum_k y_k * \frac {1}{p_k} *\frac {\partial { p_k}}{\partial z_i}
 \\ \\
  =-\sum_k y_k * \frac {1}{p_k} * p_i(\delta_{ij} -p_j) 
 \\ \\ \text{these i and j are dummy indices and we can rewrite  this as} 
\\ \\
=-\sum_k y_k * \frac {1}{p_k} * p_k(\delta_{ik} -p_i) 
\\ \\ \text{taking the two cases and adding in above equation } \\ \\
 \delta_{ij} = 1 \text{ when i =k} \text{ and } 
   \delta_{ij} = 0 \text{ when i} \ne \text{k}
   \\ \\
   = [- \sum_i y_i * \frac {1}{p_i} * p_i(1 -p_i)]+[-\sum_{k \ne i}  y_k * \frac {1}{p_k} * p_k(0 -p_i) ]
    \\ \\
     = [- y_i * \frac {1}{p_i} * p_i(1 -p_i)]+[-\sum_{k \ne i}  y_k * \frac {1}{p_k} * p_k(0 -p_i) ]
  \\ \\
     = [- y_i(1 -p_i)]+[-\sum_{k \ne i}  y_k *(0 -p_i) ]
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
\frac {\partial L}{\partial w^l} 
=  \color{red}{\frac {\partial L}{\partial z^l}}.\color{green}{\frac {\partial z^l}{\partial w^l}}
\\ \\
z^{l} = (w^l a^{l-1}+b^l) 
\\
 \color{green}{\frac {\partial z^l}{\partial w^l} = a^{l-1}}
 \\ \\ \text{Putting all together} \\ \\
 \frac {\partial L}{\partial w^l} = (p_i - y_i) *a^{l-1} \quad  \rightarrow \quad \mathbf  {EqA1}
\end{aligned}
$$

## Gradient descent

Using Gradient descent we can keep adjusting the last layer like

$$
     w{^l}{_i} = w{^l}{_i} -\alpha *  \frac {\partial L}{\partial w^l} 
$$

Now let's do the derivation for the inner layers

## Derivative of Loss with respect to Weight in Inner Layers

Adding the diagram once more 

$$
\mathbf {
\bbox[10px, border:2px solid red] { \color{red}{
\begin{aligned}
 a^0 \rightarrow
    \bbox[5px, border:2px solid black]  {
      \underbrace{\text{hidden layers}}_{a^{l-2}} }
      \,\rightarrow
    \bbox[5px, border:2px solid black]  {  
      \underbrace{w^{l-1} a^{l-2}+b^{l-1}}_{z^{l-1} }
    }
      \,\rightarrow
    \bbox[5px, border:2px solid black]  {  
      \underbrace{\sigma(z^{l-1})}_{a^{l-1}}
    }
    \,\rightarrow
    \bbox[5px, border:2px solid black]  {  
     \underbrace{w^l a^{l-1}+b^l}_{z^{l}/logits }
    }
    \,\rightarrow
    \bbox[5px, border:2px solid black]  {  
    \underbrace{P(z^l)}_{\vec P/ \text{softmax} /a^{l}}
    }
    \,\rightarrow
  \bbox[5px, border:2px solid black]  {  
    \underbrace{L ( \vec P, \vec Y)}_{\text{CrossEntropyLoss}}
  }
\end{aligned}
}}}
$$

The trick here is to find the derivative of the Loss with respect to the inner layer as a composition of the partial derivative we computed earlier. And also to compose each partial derivative as partial derivative with respect to either $z^x$ or $w^x$ but not with respect to $a^x$. This is to make derivatives easier and intuitive to compute.


$$
\begin{aligned}
\frac {\partial L}{\partial w^{l-1}} 
=  \color{blue}{\frac {\partial L}{\partial z^{l-1}}}.
     \color{green}{\frac {\partial z^{l-1}}{\partial w^{l-1}}} \rightarrow \text{EqA2}
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
\color{blue}{\frac {\partial L}{\partial z^{l-1}}} =\color{blue}{(p_i- y_i).w^l.\sigma }\color{red}{'} (z^{l-1} ) \rightarrow \text{EqA2.1 }
\\ \\
 z^{l-1} = w^{l-1} a^{l-2}+b^{l-1}
    \text{ which makes }
    \color{green}{\frac {\partial z^{l-1}}{\partial w^{l-1}}=a^{l-2}}
\\ \\
\frac {\partial L}{\partial w^{l-1}} 
=  \color{blue}{\frac {\partial L}{\partial z^{l-1}}}.
     \color{green}{\frac {\partial z^{l-1}}{\partial w^{l-1}}} = \color{blue}{(p_i- y_i).w^l.\sigma '(z^{l-1} )}.\color{green}{a^{l-2}}
\end{aligned}
$$

**Note** All the other layers should use the previously calculated value of  $\color{blue}{\frac {\partial L}{\partial z^{l-i}}}$ where  $i= current layer-1$

$$
\begin{aligned}
\frac {\partial L}{\partial w^{l-2}} 
=  \color{blue}{\frac {\partial L}{\partial z^{l-2}}}.
     \color{green}{\frac {\partial z^{l-2}}{\partial w^{l-2}}}
      \  \color{red}{ \ne (p_i- y_i)}.\color{blue}{w^{l-1}.\sigma '(z^{l-2} )}.\color{green}{a^{l-3}}
\end{aligned}
$$

Repeating the steps done in EqA.1 and EqA.2 once more for better clarity

$$
\begin{aligned}
\frac {\partial L}{\partial w^{l-2}} 
=  \color{blue}{\frac {\partial L}{\partial z^{l-2}}}.
     \color{green}{\frac {\partial z^{l-2}}{\partial w^{l-2}}} 
     \\ \\ = \color{blue}{\frac {\partial L}{\partial z^{l-1}}}.
    \frac {\partial z^{l-1}}{\partial a^{l-2}}.
    \frac {\partial a^{l-2}}{\partial z^{l-2}}
     .\color{green}{a^{l-3}}
     \\ \\
      = \color{blue}{ {\frac {\partial L}{\partial z^{l-1}}}.w^{l-1}.\sigma '(z^{l-2})}
     .\color{green}{a^{l-3}}
    \\   ( {\frac {\partial L}{\partial z^{l-1}}} \text{ calculated from previous layer})
\end{aligned}
$$

## Implementation in Python

Here is an implementation of a relatively simple Convolutional Neural Network to test out the forward and back-propagation algorithms given above [https://github.com/alexcpn/cnn_in_python](https://github.com/alexcpn/cnn_in_python). The code is well commented and you will be able to follow the forward and backward propagation with the equations above. Note that the full learning cycle is not completed; but rather a few Convolutional layers, forward propagation and backward propogation for last few layers.

## Gradient descent

Using Gradient descent we can keep adjusting the inner layers like

$$
     w{^{l-1}}{_i} = w{^{l-1}}{_i} -\alpha *  \frac {\partial L}{\partial w^{l-1}} 
$$


## Some Implementation details

Feel free to skip this section. These are some doubts that can come during implementation,and can be refereed to if needed.

**From Index Notation to Matrix Notation**

The above equations are correct only as far as the index notation is concerned. But practically we work with Weight matrices, and for that we need to write this Equation in *Matrix Notation*. For that some of the terms becomes Transposes, some matrix multiplication (dot product style) and some Hadamard product. ($\odot$). This is illustrated and commented in the code and deviates from the equations as is,

Example 
$$
\frac{\partial z^2}{\partial w^2} = (a^{1})^T 
$$


**The Jacobian Matrix**

For an input vector $\textbf{x} = \{x_1, x_2, \dots, x_n\}$ on which an element wise function is applied; say the activation function sigmoid $\sigma$; and it give the output vector $\textbf{a} = \{a_1, a_2, \dots, a_n\}$ 

$a_i= f(x_i); \text{ what is } \frac { \partial a}{ \partial x} $

In scalar case this becomes   $\frac { \partial f(x)}{ \partial x} = f'(x)$

In Vector case, that is when we take the derivative of a vector with respect to another vector to get the following (square) Jacobian matrix

Example from [ref 2]

$$
\begin{aligned}
\\ \\
\text{The Jacobain, J } = \frac {\partial a}{\partial x} = 
\begin{bmatrix}
                \frac{\partial a_{1}}{\partial x_{1}}  & \frac{\partial a_{2}}{\partial x_{1}}     & \dots     & \frac{\partial a_{n}}{\partial x_{1}}    \\
                \frac{\partial a_{1}}{\partial x_{2}}  & \frac{\partial a_{2}}{\partial x_{2}}     & \dots     & \frac{\partial a_{n}}{\partial x_{2}}    \\
                \vdots  & \vdots    & \ddots    & \vdots    \\
                \frac{\partial a_{1}}{\partial x_{n}}  & \frac{\partial a_{2}}{\partial x_{n}}    & \dots     & \frac{\partial a_{n}}{\partial x_{n}}    \\ 
\end{bmatrix}
\end{aligned}
$$

The diagonal of J are the only terms that can be nonzero:

$$
\begin{aligned}
J = \begin{bmatrix}
                \frac{\partial a_{1}}{\partial x_{1}}  & 0     & \dots     & 0    \\
                0  & \frac{\partial a_{2}}{\partial x_{2}}     & \dots     & 0    \\
                \vdots  & \vdots    & \ddots    & \vdots    \\
                0  & 0    & \dots     & \frac{\partial a_{n}}{\partial x_{n}}    \\ 
        \end{bmatrix}
\end{aligned}
$$

$$
\text{ As } 
(\frac{\partial a}{\partial x})_{ij} = \frac{\partial a_i}{\partial x_j} = \frac { \partial f(x_i)}{ \partial x_j} = 
\begin{cases}
f'(x_i)  & \text{if $i=j$} \\
0 & \text{otherwise}
\end{cases}
$$
And the authors go on to explain that $\frac{\partial a}{\partial x}$ can be written as $\text{diag}(f'(x))$ and the Hadamard or element-wise multiplication ($\odot$ or $\circ$)  can be applied instead of matrix multiplication to this Jacobian matrix like $\odot f'(x)$ when applying the Chain Rule and converting from index notation to matrix notation.

However,while implementing the neural network practically the input is not a **Vector** but an $M*N$ dimensional **Matrix** ; $M, N > 1$.

Taking a simple $2\*2$ input matrix on which the sigmoid activation function is done; the Jacobian of the same is a $8*2$ matrix and no longer a square matrix.

Does it make sense to say the derivative of Matrix $a_{i,j}$ - where an element-wise function is applied; over the input matrix $x_{i,j}$ as a Jacobian ?

$$
\frac{\partial a_{i,j}}{\partial x_{i,j}} = J_{k,l} 
$$

 There is no certainty that this will be a square matrix and we can generalize to the diagonal ? 

 However, all articles treat this matrix case as a generalization of the Vector case and write $\frac{\partial a}{\partial x}$ as the $\text{diag}(f'(x))$, and then use the element-wise/Hadamard product for the Chain Rule. This way also in implementation. But there is no meaning of diagonal in a non-square matrix. 

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

---

Hence  $\frac{\partial a_{ {i}{j}}}{\partial X}$ can be written as $\text{ diag}(f'(X))$ ; $(A =f(X))$


Note that Multiplication of a vector by a diagonal matrix is element-wise multiplication or the Hadamard product; *And matrices in Deep Learning implementation can be seen as stacked vectors for simplification.*

More details about this here [Jacobian Matrix for Element wise Opeation on a Matrix (not Vector)](https://math.stackexchange.com/questions/4397390/jacobian-matrix-of-an-element-wise-operation-on-a-matrix)




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

  [Python Impmementation of Jacobian of Softmax with respect to Logits Aerin Kim]: https://stackoverflow.com/a/46028029/429476
  
  [Derivative of Softmax Activation -Alijah Ahmed]: https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
  
  [Dertivative of SoftMax Antoni Parellada]: https://stats.stackexchange.com/a/267988/191675

  [Backpropagation In Convolutional Neural Networks Jefkine]: https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/]

  [Finding the Cost Function of Neural Networks Chi-Feng Wang]: https://towardsdatascience.com/step-by-step-the-math-behind-neural-networks-490dc1f3cfd9

  [Vector Derivatives]: http://cs231n.stanford.edu/vecDerivs.pdf
 
 ---
 End