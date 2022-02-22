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

## Chapter 7: Backpropagation with Softmax and  Cross Entropy Loss 

---

Before we start it is better to be clear of the model of the neural network we are discussing here.

Repeating from Chapter 4

Let's  think of a $l$ layered neural network whose input is $x/a^0$ and output is $a^l$

$$
 x/a^0 \rightarrow \text{hidden layers} \,\rightarrow a^{l-1} \rightarrow  a^{l} 
$$

$a^{l}$  is the input at layer *l*. Input  of a neuron in layer *l*  is the output of activation from the previous layer $(l-1)$.

Output of Activation is  the product of the weight *w* and input at layer $(l-1)$  plus the *basis*, passed to the *activation function*. 
Writing this below gives.

$$
\begin{aligned}
  a^{l} = ActivationFunction(w^l a^{l-1}+b^l).
\\ \\
  a^{l} = ActivationFunction(z^l).
\\ \\
\text{where} \quad w^l a^{l-1}+b^l = z^{l}
\end{aligned}
$$


In this network we will be using the **sigmoid ($\sigma$ )** function as the activation function for all layers except the last layer $l$. and for the last layer we use the **Softmax activation function**. We will use the **Cross Entropy Loss** as the loss function. 

Writing this out, without index notation, and with the super script representing just the layers of the network.

Please take few minutes to read this short article about index notation [A Primer on Index Notation John Crimaldi]

### The Network

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
     \underbrace{w^l a^{l-1}+b^l}_{z^{l} }
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

 $Y$ is the target vector or the Truth vector. This is a one hot encoded vector, example  $Y=[0,1,0]$, here the second element is the desired class.The training is done so that the CrossEntropyLoss is minimised using Gradient Loss algorithm.

$P$ is the Softmax output and is the activation of the last layer $a^l$. This is a vector. All elements of the Softmax output add to 1; hence this is a probability distribution unlike a Sigmoid output.The Cross Entropy Loss $L$ is a Scalar.

Note the Index notation is the represention an element of a Vector or a Tensor, and is easier to deal with while deriving out the equations.

**Softmax** (in Index notation)

Below I am skipping the superscript part, which I used to represent the layers of the network.

$$
\begin{aligned}
p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}
\end{aligned}
$$

This represent one element of the softmax vector, example $\vec P= [p_1,p_2,p_3]$

**Cross Entropy Loss** (in Index notation)

Here $y_i$ is the indexed notation of an element in the target vector  $Y$. T

$$
\begin{aligned}
L = -\sum_j y_j \log p_j
\end{aligned}
$$


---
## YABE - Yet Another Back-propogation Explanation - this time for Dummies

Our aim is to adjust the Weight matrices in all the layers so that the Softmax output reflects the Target Vector for the set of training inputs.

For this we need to find the derivative of Loss function wrto weights, that is to get the gradient of the loss function wrto weight. We then use that gradient matrix to minimise the Loss function iteratively using the gradient descent algorithm.

This is a bit involved mathematically and various authors give various ways. There are many terms and concepts that can trip somone who has not touched maths for sometime or done these parts or paid specific attention to these parts.

However there are only a few concepts that you need to understand; 

- Understand what Scalar's, Vectors, Tensors are and that Vectors and Tensors are written as matrices and Vector is one dimenstion matrix whereas Tensor's are many dimensional usually. (Technically a Vector is also a Tensor). After this you can forget about Tensors and think only of Vectors and Matrices and Scalar. The only Scalar would be the output of the Loss funcion.

- That linear algebra for matrices that will be used is just poperties of matrix multiplication and addition that you already know. A linear equation of the form $y= m*x +c$ in matrix form used in neural network is $z_l = w_l* a_{l-1} + b_l$. 
  - Onlt if two matrices are Vectors then matrix multiplication is called a dot product.However each row of a multidimenstional matrix acts like a Vector and the Numpy dot function(numpy.dot) is generally used in most examples for matrix multiplication; and it works athe same. Though technically numpy matmul is the right one to use.$np.dot(A,B)$ is same as $np.matmul(A,B)$. Note also that numpy.einsum is also used for dimensions more than two.There is no accepetd definition of matrix multiplication of dimensions other than two! 
  If A and B are two dimenstional matrices $np.dot(A,B) = np.einsum('ij,jk->ik', A, B)$. And einsum is much easier than numpy.tensordot to work with. You will need this for higher dimensional weights especially trying to implement convolutional neural network. Using the Einstein summation convention, many common multi-dimensional, linear algebraic array operations can be represented in a simple fashion -https://numpy.org/doc/stable/reference/generated/numpy.einsum.html*
  - Hadamard product you rarely if ever use or need, though internally computer implementation maybe using these for optimisation. it is a special case of element wise multiplication of matrices of same dimension and used in multiplying the gradient vector.It is reffered in Michel Neilsen famous book in writing out the Error of a layer wrto pervious layers. http://neuralnetworksanddeeplearning.com/chap2.html.

- That you don't need to derive the Jacobian matrix for this. Though just for understanding this is needed and you need to see that. Why this is not needed ? See below

- That many authors mean many things when talking about Chain Rule. Single variable, MultiVariable Chain rule etc and how /todo.



Expand the understanding that derivative of a function stands for slope of the function to the general concept that derivative of a function gives the gradient of the function





Please check the reference for the popular articles and what I reffered.

 What I did was also to code a CNN with Softmax and CrossEntropy and tried to fit the equations to code. And the one that worked for me  finally (at least till the weight dimension part) is this [Supervised Deep Learning Marc'Aurelio Ranzato DeepMind].

 **Note about Matrix Calulus/Vector Calculus/ Jacobian Matrix**
 
 You do not need to derive the Jacobian matrix to find the gradients for backpropogation, while implementing. 

*So it's entirely possible to compute the derivative of the softmax layer without actual Jacobian matrix multiplication; and that's good, because matrix multiplication is expensive! The reason we can avoid most computation is that the Jacobian of the fully-connected layer is sparse.- [The Softmax function and its derivative-Eli Bendersky]*

But Eli Bendersky's above page is something that can help you understand in depth for the softmax derivative wrto its weights.

That said, when you convert from Index notation to actual matrix notation, for example for implementation then you will need to understand how the index multiplication transforms to Matrix multiplication and transpose.

Example from [The Matrix Calculus You Need For Deep Learning (Derivative wrto Bias) Terence,Jermy]

$$
\frac{\partial z^2}{\partial w^2} = (1^{\rightarrow})^T* diag(a^1) =(a^{1})^T \quad
$$

 **Note about Chain Rule**

The Chain rule is heavily used in BackPropogation for re-writing the equations using terms in the inner layers, to make the derivative's possible.

The single variable chain rule is simple 
## For the last layer

We will follow [Supervised Deep Learning Marc'Aurelio Ranzato DeepMind] for derivind for the last layer as well as the inner layer, with some help from answers in stackoverflow and related sites. Most of the information is duplicated from the reference, but unless one derives it, it's easy to gloss over and not understand how its done.


$$
\mathbf {
\frac {\partial L}{\partial w^l} 
=  \color{red}{\frac {\partial L}{\partial z^l}}.\color{green}{\frac {\partial z^l}{\partial w^l}}
}
$$

Where
$$
\mathbf {
L = -\sum_k y_k \log \color{red}{p_k} \,\,and \,p_j = \frac {e^ \color{red}{z_j}} {\sum_k e^{z_k}}
}
$$
If you are confused with the indexes, just take a short example and substitute. Basically i,j,k etc are dummy indices used to illustrate in index notation the vectors.

I am going to drop the superscirpt $l$ denoting the layer number henceforth and focus on the index notation for the softmax vector $P$ and target vector $Y$

Following from [Derivative of Softmax Activation -Alijah Ahmed]

$$ \color{red}
  {
  \begin{aligned}

    \frac {\partial L}{\partial z_i} = \frac {\partial ({-\sum_j y_k \log {p_k})}}{\partial z_i}
   \\ \\ \text {taking the summation outside} \\ \\
   = -\sum_j y_k\frac {\partial ({ \log {p_k})}}{\partial z_i}
  \\ \\ \text {using the derivative of the logarithm} \\ \\
  = -\sum_k y_k * \frac {1}{p_k} *\frac {\partial { p_k}}{\partial z_i}
  
\end{aligned}
}
$$

The last term is the derivative  of Softmax wrto it's inputs, this is easy to derive and there are many sites that descirbe it. Example [Neural Network with SoftMax in Python- Abishek Jana] or the first part here  [The Softmax function and its derivative-Eli Bendersky]

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

Using this above

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
     \frac {\partial L}{\partial z^l}  = p_i - y_i
\end{aligned}
}
$$

Note, that we now need to calculate the second term, to complete the equation

$$
\begin{aligned}
\frac {\partial L}{\partial w^l} 
=  \color{red}{\frac {\partial L}{\partial z^l}}.\color{green}{\frac {\partial z^l}{\partial w^l}}
\\ \\
z^{l} = (w^l a^{l-1}+b^l) 
\\
 \color{green}{\frac {\partial z^l}{\partial w^l} = a^{l-1}}
 \\ \\ \text{Putting all together} \\ \\
 \frac {\partial L}{\partial w^l} = (p_i - y_i) *a^{l-1} \quad  \quad(\mathbf  {1})
\end{aligned}
$$


Using Gradient descent we can keep adjusting the last layers like

$$
     w{^l}{_i} = w{^l}{_i} -\alpha *  \frac {\partial L}{\partial w^l} 
$$

Now let's do the derivation for the inner layers

## For the inner layers

Note that in some sites you may see this definition below, which is wrong as it wont work out for $l-2$ and other layers.

$$
  \begin{aligned}
 \require{cancel}\cancel{
\frac {\partial L}{\partial w^{l-1}} 
=  \color{red}{\frac {\partial L}{\partial z^l}}.
    \color{blue}{\frac {\partial z^l}{\partial a^{l-1}}}.
    \color{violet}{\frac {\partial a^{l-1}}{\partial z^{l-1}}}.
    \color{green}{\frac {\partial z^{l-1}}{\partial w^{l-1}}}
\\ \\
\frac {\partial L}{\partial w^{l-1}} 
= \mathbf  (a^l-y).w^l.\sigma'(z^{l-1}).a^{l-2} \quad  \quad {(2)}
}
\end{aligned}
$$


Using Gradient descent we can keep adjusting the inner layers like

$$
     w{^{l-1}}{_i} = w{^{l-1}}{_i} -\alpha *  \frac {\partial L}{\partial w^{l-1}} 
$$


## References
 
 (Perspective from a Programmer who has some memory of Calculus and willing to learn more)
 
 Easier to follow (wihtout explcit Matrix Calculus) and something that can be implemented
 - [Supervised Deep Learning Marc'Aurelio Ranzato DeepMind]  

Easy to follow but lacking in some aspects
- [Notes on Backpropagation-Peter Sadowski]

Slightly hard to follow using the Jacobian - but partly implementable
 - [The Softmax function and its derivative-Eli Bendersky]
 - [Derivative of Softmax Activation -Alijah Ahmed]

 Please also check [The Matrix Calculus You Need For Deep Learning Terence,Jermy] for the above and more exlanation [Finding the Cost Function of Neural Networks Chi-Feng Wang]

Easy to follow but lacking in rigour and not correct for all types of networks
 - [Neural Network with SoftMax in Python- Abishek Jana]
 
More difficult to Follow with proper index notations (I could not)
 - [Backpropagation In Convolutional Neural Networks Jefkine]

  
  [Supervised Deep Learning Marc'Aurelio Ranzato DeepMind]: https://bfeba431-a-62cb3a1a-s-sites.googlegroups.com/site/deeplearningcvpr2014/ranzato_cvpr2014_DLtutorial.pdf?attachauth=ANoY7cqPhkgQyNhJ9E7rmSk-RTdMYSYqpfJU2gPlb9cWH_4a1MbiYPq_0ihyuolPiYDkImyr9PmA-QwSuS8F3OMChiF97XTDD_luJqam70GvAC4X6G6KlU2r7Pv1rqkHaMbmXpdtXJHAveR_jWf1my_IojxFact87u2-1YXtfJIwYkhBwhMsYagICk-P6X9ktA0Pyjd601tboSlX_UGftX1vB57-tS6bdAkukhmSRLU-ZiF4RdJ_sI3YAGaaPYj1KLWFpkFa_-XG&attredirects=1
  
  [lecun-ranzato]: https://cs.nyu.edu/~yann/talks/lecun-ranzato-icml2013.pdf
  
  [Euclidean_vector]: https://en.wikipedia.org/wiki/Euclidean_vector
  
  [A Primer on Index Notation John Crimaldi]: https://web.iitd.ac.in/~pmvs/courses/mcl702/notation.pdf
  
  [backpropogationgif]: https://i.imgur.com/jQOLUG3.gif
  [Notes on Backpropagation-Peter Sadowski]: https://www.ics.uci.edu/~pjsadows/notes.pdf
  
  [The Softmax function and its derivative-Eli Bendersky]: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  
  [Neural Network with SoftMax in Python- Abishek Jana]: https://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
  
  [Derivative of Softmax Activation -Alijah Ahmed]: https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
  
  [Backpropagation In Convolutional Neural Networks Jefkine]: https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/]

  [Finding the Cost Function of Neural Networks Chi-Feng Wang]: https://towardsdatascience.com/step-by-step-the-math-behind-neural-networks-490dc1f3cfd9
  
  [The Matrix Calculus You Need For Deep Learning Terence,Jermy]:https://arxiv.org/pdf/1802.01528.pdf

  [The Matrix Calculus You Need For Deep Learning (Derivative wrto Bias) Terence,Jermy]: https://explained.ai/matrix-calculus/#sec6.2

