# The Maths behind Neural Networks

Alex Punnen \
&copy; All Rights Reserved \
2019-2021 

---

## Contents

- Chapter 1: [Vectors, Dot Products and  the Perceptron](1_vectors_dot_product_and_perceptron.md)
- Chapter 2: [Feature Vectors, Dot products and Perceptron Training](2_perceptron_training.md)
- **Chapter 3: [Gradient Descent, Gradient Vector and Loss Function](3_gradient_descent.md)**
- Chapter 4: [Activation functions, Cost functions and Back propagation](4_backpropogation.md)
- Chapter 5: [Back Propagation with Matrix Calulus](5_backpropogation_matrix_calulus.md)
- Chapter 6: [A Simple NeuralNet with above Equations](6_neuralnetworkimpementation.md)
- Chapter 7: [Back Propagation for Softmax with CrossEntropy Loss](7_cnn_network.md)

## Chapter 7: Back Propagation for Softmax with CrossEntropy Loss

Let's think of a $l$ layered neural network whose input is $x=a^0$ and output is $a^l$.In this network we will be using the **sigmoid ($\sigma$ )** function as the activation function for all layers except the last layer $l$. For the last layer we use the **Softmax activation function**. We will use the **Cross Entropy Loss** as the loss function.

---

Our aim is to adjust the Weight matrices in all the layers so that the Softmax output reflects the Target Vector for the set of training inputs.

For this we need to find the derivative of Loss function with respect to weights, that is to get the gradient of the loss function with respect to weight. We then use that gradient matrix to minimize the Loss function iteratively using the gradient descent algorithm.

This is a bit involved mathematically and various authors give various ways. There are many terms and concepts that can trip someone who has not touched maths for some time or done these parts or paid specific attention to these parts.

# The Maths you Need for Back Propogation
## and the Maths you probably don't 

Truth is that to understand this correctly, as in dealing with  higher dimensional weight matrices, you need a good understanding of index conventions when writing Matrices, and how these translate to Matrices. Also, a few other topics are given below. However, with some understanding, you can understand the concepts easily. Also many do not bother to understand at all, as we have beautiful Keras or Tensorflow abstractions. But to try to improve, this understanding would be necessary.

There is a vast amount of articles and explanations, I will be referring mostly to a few highly quoted articles here. The rest are all quoting from these few.

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
 
- Caluculus - **Chain Rule - Single variable, Multi variable Chain rule, Vector Chain Rule**
 
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

The trick here (yes it is a trick), is to derivative the Loss with respect to the inner layer as a composition of the partial derivative we computed earlier. And also to compose each partial derivative as partial derivative with respect to either $z^x$ or $w^x$ but not with respect to $a^x$. This is to make derivatives easier and intuitive to compute.


$$
\begin{aligned}
\frac {\partial L}{\partial w^{l-1}} 
=  \color{blue}{\frac {\partial L}{\partial z^{l-1}}}.
     \color{green}{\frac {\partial z^{l-1}}{\partial w^{l-1}}} \rightarrow \text{EqA2}
\end{aligned}
$$

The trick is to represent the first  part  in terms of what we computed earlier; in terms of $\color{blue}{\frac {\partial L}{\partial z^{l}}}$

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
\color{blue}{\frac {\partial L}{\partial z^{l-1}}} =\color{blue}{(p_i- y_i)}.w^l.\sigma \color{red}{'} (z^{l-1} ) \rightarrow \text{EqA2.1 }
\\ \\
 z^{l-1} = w^{l-1} a^{l-2}+b^{l-1}
    \text{ which makes }
    \color{green}{\frac {\partial z^{l-1}}{\partial w^{l-1}}=a^{l-2}}
\\ \\
\frac {\partial L}{\partial w^{l-1}} 
=  \color{blue}{\frac {\partial L}{\partial z^{l-1}}}.
     \color{green}{\frac {\partial z^{l-1}}{\partial w^{l-1}}} = \color{blue}{(p_i- y_i)}.w^l.\sigma \color{red}{'} (z^{l-1} ).\color{green}{a^{l-2}}
\end{aligned}
$$

**Note** The value of EqA2.1 to be used in the next layer derivations; and repeated to the first layer; ie do not repeat $(p_i -y_i)$

$$
\begin{aligned}
\frac {\partial L}{\partial w^{l-2}} 
=  \color{blue}{\frac {\partial L}{\partial z^{l-2}}}.
     \color{green}{\frac {\partial z^{l-2}}{\partial w^{l-2}}} \color{red}{ \ne (p_i- y_i).w^{l-1}.\sigma '(z^{l-2} ).a^{l-3}}
\end{aligned}
$$

**Disclaimer**
As proud I am to understand till here, I had the misfortune of trying to implement a CNN with this and I know what is all **wrong** with the above.Basically if you use the above equation, you will find that the weights do not match for matrix multiplication.

This is because, the above equation is correct only as far as the index notation is concerned. But practically we work with weight matrices, and for that we need to write this Equation in Matrix Notation. For that some of the terms becomes Transposes, some matrix multiplication (dot product style) and some Hadamard product. ($\odot$). All these are detailed out in [The Matrix Calculus You Need For Deep Learning Terence,Jermy], and I need to edit the answer and explanation once I have grasped if with an example. Only then will the weight dimension align correctly. Please see the correct equations here [Neural Networks and Deep Learning Michel Neilsen]

Example 
$$
\frac{\partial z^2}{\partial w^2} = (a^{1})^T 
$$

## Gradient descent

Using Gradient descent we can keep adjusting the inner layers like

$$
     w{^{l-1}}{_i} = w{^{l-1}}{_i} -\alpha *  \frac {\partial L}{\partial w^{l-1}} 
$$


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
 
