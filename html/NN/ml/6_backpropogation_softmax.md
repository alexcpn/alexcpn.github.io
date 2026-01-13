# The Maths of Deep Learning

Alex Punnen \
&copy; All Rights Reserved


## Backpropagation with Softmax and Cross Entropy


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





## Some Implementation details

For a detailed explanation of the Matrix Calculus, Jacobian, and Hadamard product used here, please refer to **Chapter 6: Back Propagation - Matrix Calculus**.

**From Index Notation to Matrix Notation**

The equations above use index notation for clarity. In practice, we use Matrix Notation which involves Transposes and Hadamard products as explained in the previous chapter.

### Implementation in Python

Here is an implementation of a relatively simple Convolutional Neural Network to test out the forward and back-propagation algorithms given above [https://github.com/alexcpn/cnn_in_python](https://github.com/alexcpn/cnn_in_python). The code is well commented and you will be able to follow the forward and backward propagation with the equations above.

## Gradient descent

Using Gradient descent we can keep adjusting the inner layers like

$$
     W{^{l-1}} = W{^{l-1}} -\alpha \cdot \frac {\partial L}{\partial W^{l-1}} 
$$ 

Next: [Neural Network Implementation](7_neuralnetworkimpementation.md)

## References
 
- [Supervised Deep Learning - Marc'Aurelio Ranzato (DeepMind)](https://bfeba431-a-62cb3a1a-s-sites.googlegroups.com/site/deeplearningcvpr2014/ranzato_cvpr2014_DLtutorial.pdf?attachauth=ANoY7cqPhkgQyNhJ9E7rmSk-RTdMYSYqpfJU2gPlb9cWH_4a1MbiYPq_0ihyuolPiYDkImyr9PmA-QwSuS8F3OMChiF97XTDD_luJqam70GvAC4X6G6KlU2r7Pv1rqkHaMbmXpdtXJHAveR_jWf1my_IojxFact87u2-1YXtfJIwYkhBwhMsYagICk-P6X9ktA0Pyjd601tboSlX_UGftX1vB57-tS6bdAkukhmSRLU-ZiF4RdJ_sI3YAGaaPYj1KLWFpkFa_-XG&attredirects=1) - Easier to follow (without explicit Matrix Calculus) though not really correct
- [Notes on Backpropagation - Peter Sadowski](https://www.ics.uci.edu/~pjsadows/notes.pdf) - Easy to follow but lacking in some aspects
- [The Softmax function and its derivative - Eli Bendersky](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/) - Slightly hard to follow using the Jacobian
- [Backpropagation In Convolutional Neural Networks - Jefkine](https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/]) - More difficult to follow with proper index notations (I could not) and probably correct
- [A Primer on Index Notation - John Crimaldi](https://web.iitd.ac.in/~pmvs/courses/mcl702/notation.pdf)
- [The Matrix Calculus You Need For Deep Learning - Terence Parr & Jeremy Howard](https://arxiv.org/pdf/1802.01528.pdf)
- [Neural Networks and Deep Learning - Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap2.html)
- [Derivative of Softmax Activation - Alijah Ahmed](https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function)


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
 



