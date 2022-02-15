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

Before we start; for non maths people a quick reminder of matrices,vectors and tensors and the notations that we will use.

### Matrices - A way to represent Vectors (and Tensors)

 Vectors are represented as matrices.A matrix is defined to be a rectangular array of numbers. Example here is a [Euclidean Vector][Euclidean_vector]  in three-dimensional Euclidean space (or $R^{3}$) with some magnitude and direction (from (0,0,0) origin in this case).
 
 A vector is represented either as column matrix or as a row matrix.

$$
a = \begin{bmatrix}
a_{1}\\a_{2}\\a_{3}\ 
\end{bmatrix} = \begin{bmatrix} a_{1} & a_{2} &a_{3}\end{bmatrix}
$$

$a_{1},a_{2},a_{3}$ are the component scalars of the vector. A vector is represented as $\vec a$ in the **Vector notation** and as $a_{i}$ in the **Index Notation**. 

Please see [this article][indexnotation] regarding index notation details and about *free indices*. In most of the derivations later on, we will use the index notation.

**Tensor**

Since  we will be dealing soon with multidimensonal matrices, it is as well to state here what Tensors are. Easier is to define how they are represented, and that will suit our case as well. A Vector is a one dimensional matrix. Higher dimension matrices are used for Tensors. Example is the multidimensional weight matrices we use in neural network. They are weight Tensors. A Vector is a Tensor or Rank 1 and technically a Scalar is also a Tensor of Rank 0.

---

Before we start it is better to be clear of the model of the neural network we are discussing here.

Repeating from Chapter 4


 Let us see the mathematical notation of representing  a neural network.

 A real neural network has many layers and many interconnections. Let's  think of a $l$ layered neural network with just a linear connection between two layers, l and (l-1) to simplify the notation and to understand the maths.

$$
 x \rightarrow \text{hidden layers} \,\rightarrow a^{l-1} \rightarrow  a^{l} \rightarrow  output
 $$

   We take  $a^{l}$  as the input at layer *l*. Input  of a neuron in layer *l*  is the output of activation from the previous layer $(l-1)$.

 Output of Activation is  the product of the weight *w* and input at layer $(l-1)$  plus the *basis*, passed to the *activation function*. 
Writing this below gives.

$$
  a^{l} = ActivationFunction(w^l a^{l-1}+b^l).
$$

Note the notation we use $z^{l}$ which we will come to in a moment to see how it is used

$$
  z^{l} = (w^l a^{l-1}+b^l).
$$

So

$$
  a^{l} = ActivationFunction(z^l).
$$

For this case we will be using the sigmoid ($\sigma$ ) function as the activation function for all layers except the last layer- $l$, and for the last layer we use the Softmax activation function. 

Writing this out, without index notation, and with the super script representing just the layers of the network

$$
\begin{aligned}
 x \rightarrow \text{hidden layers} \,\rightarrow (w^{l-1} a^{l-2}+b^{l-1})=z^{l-1} \\
 z^{l-1} \rightarrow  \sigma(z^{l-1})= a^{l-1} \rightarrow  (w^l a^{l-1}+b^l)=z^{l} 
\\
P(z^l) = a^{l}
\\
P(z^l),Y \rightarrow CrossEntropyLoss(P(z^l),Y) =L
\end{aligned}
$$

 $Y$ is the target vector. This is a one hot encoded vector like $[0,1,0]$, here the second element is the desired class and the training is done so that the CrossEntropyLoss is minimised using Gradient Loss algorithm.


$P$ is the Softmax output and is the activation of the last layer $a^l$. This is a vector. All elements of the Softmax output add to 1; and hence this is a probability distribution unlike a Sigmoid output.

The Cross Entropy Loss $L$ is a Scalar.

To quickly write out the equations of these

**Softmax** in Index notation. 

Below I am skipping the superscript,but this is for the layer $l$ --> $z^l$.

This represent one element of the softmax vector, say something like $[p_1,p_2,p_3]$
$$
p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}
$$

**Cross Entropy Loss** in Index notation

Here $y_i$ is the indexed notation of an element in the target vector  $Y$. This is a one hot encoded vector like $[0,1,0]$

$$
L = -\sum_j y_j \log p_j
$$

---
## Back Propogation

Our aim is to adjust the Weight matrices in all the layers so that the Softmax output reflects the Target out for a set of training inputs.

For this we need to find the derivative of Loss function wrto weights, to minimise the Loss function and use the gradient descent algorithm to adjust the weights iteratively.

![backpropogationgif]


## References
 
 - [Notes on Backpropagation-Peter Sadowski]
 - [The Softmax function and its derivative-Eli Bendersky]
 - [Derivative of Softmax Activation -Alijah Ahmed]
 - [Neural Network with SoftMax in Python- Abishek Jana]


  [Euclidean_vector]: https://en.wikipedia.org/wiki/Euclidean_vector
  [indexnotation]: https://web.iitd.ac.in/~pmvs/courses/mcl702/notation.pdf
  [backpropogationgif]: https://i.imgur.com/jQOLUG3.gif
  [Notes on Backpropagation-Peter Sadowski]: https://www.ics.uci.edu/~pjsadows/notes.pdf
  [The Softmax function and its derivative-Eli Bendersky]: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  [Neural Network with SoftMax in Python- Abishek Jana]: https://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
  [Derivative of Softmax Activation -Alijah Ahmed]: https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
