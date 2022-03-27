# The Maths behind Neural Networks

Alex Punnen \
&copy; All Rights Reserved \
2019-2021 

---

## Contents

### Introduction

The intention of this set of articles is mostly for self-learning and to present what is learned in a simple and connected manner so that it could help others.

The aim of this set of articles if to unravel the mathematical concepts used in Deep Learning. I have been an application programmer for almost twenty years; and for me, and people like me, is not so difficult to use some platform like Tensorflow or Keras or Pytorch and off-the shelf models or libraries to assemble an IT Deep learning solution. That is a skill, and has its place. However it will benefit the builders also, to have more intuition on the things they build.

Note that I have called out Deep Learning specifically. Deep Learning is a branch of machine learning. However machine learning, as is widely understood now means Data Science - that is data analysis using Statistics and Probability Distribution ( Frequentest and Bayesian) of a Data Set; simple example is linear regression (curve fitting) or logistic regression (hyper-plane fitting).

Deep Learning is concerned with no such statistical or probabilistic modelling; it is not a mathematical model at all. It can be thought of as an universal approximation of any model; and the approximation bettered by more training on more data. 

Another common explanation is that it models neural network in the brain. But that is an analogy that is way off the mark as of now. This complex approximation machine is modelled with the concept of vectors, matrices, using the concept of cost function similar to linear regression solutions to find a reduced cost by algorithms like gradient descent; and then propagating the cost back to all constituents of the machine via a method called back-propagation.

To further explain in more concrete terms,this distinction of Deep Learning from Machine Learning, we can take the example of object detection in images. You can use a machine learning model like HAAR or HOG for object detection im images. But what works for one kind of images will not necessarily be effective in another kind. You need to manually tune the hyper-parameters of these models. This is the classical machine learning way of object detection. Whereas in Deep Learning way, we model a special type of Neural Network (or Deep Learning model) and train it using images to automatically tune the internal parameters of the network to be able to detect the objects from a wider gamut of image scenes.Deep Learning models are much more accurate than machine learning models; and now used as defacto in all production use-cases.

Whatever be the model used in Deep Learning, the learning is via gradient descent and back-propogation as of to-date; and how that works is basically what we are trying to learn.

- **Chapter 1: [The simplest Neural Network - Perceptron using Vectors and Dot Products](1_vectors_dot_product_and_perceptron.md)**
- Chapter 2: [Perceptron Training via Feature Vectors and Dot product ](2_perceptron_training.md)
- Chapter 3: [Gradient Descent, Gradient Vector and Loss Function](3_gradient_descent.md)
- Chapter 4: [Activation functions, Cost functions and Back propagation](4_backpropogation.md)
- Chapter 5: [Back Propagation with Matrix Calulus](5_backpropogation_matrix_calulus.md)
- Chapter 6: [A Simple NeuralNet with above Equations](6_neuralnetworkimpementation.md)
- Chapter 7: [Back Propagation for Softmax with CrossEntropy Loss](7_cnn_network.md)

# Chapter 1
## The simplest Neural Network - Perceptron using Vectors and Dot Products

To understand the maths or modelling of the neural network, it is best to start from the beginning when things were simple. The earliest neural network - the `Rosenblatt’s Perceptron` was the first to introduce the concept of using vectors and the property of vector dot product, to split hyperplanes of input feature vectors. These are the fundamentals that are still used today.

Before we talk about why Neural Network inputs and weights are modelled as vectors (and represented as matrices) let us refresh our maths and see what these concepts mean geometrically. This will help us in understanding the intuition of these when they are used in other contexts/ in higher dimensions.

## Vectors

A vector is an object that has both a magnitude and a direction. Example Force and Velocity. Both have magnitude as well as direction.

However we need to specify also a context where this vector lives -[Vector Space][1]. For example when we are thinking about something like [Force vector][2], the context is usually 2D or 3D Euclidean world.

![vector2D][3]

![vector3D][4]

(Source: 3Blue1Brown)

The easiest way to understand the Vector is in such a geometric context, say 2D or 3D cartesian coordinates, and then extrapolate it for other Vector spaces which we encounter but cannot really imagine.

## Matrices - A way to represent Vectors (and Tensors)

 Vectors are represented as matrices. A Vector is a one dimensional matrix.A matrix is defined to be a rectangular array of numbers. Example here is a [Euclidean Vector][Euclidean_vector]  in three-dimensional Euclidean space (or $R^{3}$) with some magnitude and direction (from (0,0,0) origin in this case).
 
 A vector is represented either as column matrix (m*1)or as a row matrix (1*m).

$$
a = \begin{bmatrix}
a_{1}\\a_{2}\\a_{3}\ 
\end{bmatrix} = \begin{bmatrix} a_{1} & a_{2} &a_{3}\end{bmatrix}
$$

$a_{1},a_{2},a_{3}$ are the component scalars of the vector. A vector is represented as $\vec a$ in the **Vector notation** and as $a_{i}$ in the **Index Notation**. 

Two dimensional matrices can be thought of as one dimensional vectors stacked on top of each other. This intuition is especially helpful when we use dot products on neural network weight matrices.

Please see [this article][indexnotation] regarding index notation details and about *free indices*. In most of the derivations later on, we will use the index notation.

In classical machine learning, example for linear regression -(line or curve fitting methodology), matrices play an important part in representing the set of modelled linear equations in matrix form and then using the matrix properties -matrix triangulation methods, to solve the linear equations. See [matrix decomposition] methods, example LU, QR decomposition. The other option is to find the optimal solution set by gradient descent, as it is computationally efficient.

In Deep learning, the concept is not of linear regression or set of linear equations; but matrices still are heavily used to model the neural network. Here gradient descent is heavily used to find optimal solution.


**Tensor**

Since  we will be dealing soon with multidimensional matrices, it is as well to state here what Tensors are. Easier is to define how they are represented, and that will suit our case as well. We have seen that a Vector is a one dimensional matrix. Higher dimension matrices are used for Tensors. Example is the multidimensional weight matrices we use in neural network. They are weight Tensors. A Vector is a Tensor or Rank 1 and technically a Scalar is also a Tensor of Rank 0.

## Dot product

Algebraically, the dot product is the sum of the products of the corresponding entries of the two sequences of numbers.

if $\vec a = \left\langle {a_1,a_2,a_3} \right\rangle$ and $\vec b = \left\langle {b_1,b_2,b_3} \right\rangle$, then 

$\vec a \cdot \vec b = {a_1}{b_1} + {a_2}{b_2} + {a_3}{b_3} = a_ib_i \quad\text {in index notation}$

**Geometrically**, it is the product of the Euclidean magnitudes of the two vectors and the cosine of the angle between them

$$
 \vec a \cdot \vec b = \left\| {\vec a} \right\|\,\,\left\| {\vec b} \right\|\cos \theta 
$$

 ![dotproduct][6]

Note- These definitions are equivalent when using Cartesian coordinates.
Here is a simple proof that follows from trigonometry [8] and [9]

## Dot Product and Vector Alignment

**If two Vectors are in the same direction the dot product is positive and if they are in the opposite direction the dot product is negative.** This can be visualized geometrically putting in the value of the Cosine angle.

So we could use the dot product as a way to find out if two vectors are aligned or not.

## Dot Product and Splitting the hyper-plane

Imagine we have a problem of  classifying if a leaf is healthy or not based on certain features of the leaf. For each leaf we have some feature vector set.

 For any  **input feature vector** in that vector space, if we have a **weight vector**, whose dot product with one feature vector of the set of input vectors of a certain class (say leaf is healthy) is positive and with the other set is negative, then that weight vector is splitting the feature vector hyper-plane into two.
 
 **In essence, we are using the weight vectors to split the hyper-plane into two distinctive sets.**

 For any new leaf, if we only extract the same features into a feature vector; we can **dot it with the *trained* weight vector and find out if it falls in healthy or deceased class.

 Not all problems have their feature set which is linearly seperable. So this is a constraint of this system.
## Modelling of the Perceptron

The initial neural network - the **Frank Rosenblatt's perceptron** was basically splitting a linearly seperable feature set into distinct sets. Here is how the Rosenblatt's perceptron is modelled

  ![perceptron2][7]

[Image source ][11]

Inputs are $x_1$ to $x_n$ , weights are some values that are learned $w_1$ to $w_n$. There is also a bias (b)  which in above is  -$\theta$

The bias can be modelled as a a weight $w_0$ connected to a dummy input $x_0$ set to 1.

If we ignore bias term, the output $y$ can be written as the sum of all inputs times the weights; thresholded by zero.

$$
y = 1  \text{ if } \;\sum_i w_i x_i \ge 0  \text{ else } y=0
$$

The Perceptron will fire if the sum of its inputs is greater than zero; else it will not. It will be *activated* if the sum of its inputs is greater than zero; else it will not.

## The Activation Function of the Nerual Network

The big blue circle is the primitive brain of the primitive neural network - the perceptron brain. Which is basically a function $\sigma$ (sigma).

 This is what is called as an **Activation Function** in Neural Networks. We will see that later. 
 
 In Perceptron, this is a step function we use here, output is non continuous (and hence non-differentiable) and is either 1 or 0.

If the inputs are arranged as a column matrix and weights also arranged likewise then both the input and weights can be treated as vector and $\sum_i w_i x_i$ is same as the dot product $\mathbf{w}\cdot\mathbf{x}$. 

Hence the activation function can also be written as 

$$
\sigma (x) =
\begin{cases}
1, & \text{if}\ \mathbf{w}\cdot\mathbf{x}+b \ge 0 \\
0, & \text{otherwise} 
\end{cases}
$$

Note that dot product of two matrices (representing vectors), can be written as that transpose of one multiplied by another, $w \cdot x = w^Tx$ 

$$
\sigma(w^Tx + b)=
\begin{cases}
1, & \text{if}\ w^Tx + b \ge 0 \\
0, & \text{otherwise} 
\end{cases}
$$

All three equations are the same, just that in different reference it will be written in one of these forms.

### The equation $w \cdot x \gt b$  defines all the points on one side of the hyperplane, and $w \cdot x \ge b$  all the points on the other side of  the hyperplane and on the hyperplane itself.

### This happens to be the  very definition of “linear separability”

### **Thus, the perceptron allows us to separate our feature space in two convex half-spaces**
Ref ([12])

If we can calculate or train the weight matrix used here, then we can have a weight vector, which splits the input feature vectors to two regions by a hyperplane. 

**This is the essence of the Perceptron, the initial artificial neuron.**

![hyperplane1]

[Image source][13]

In simple terms, it means that an unknown feature vector of an input set belonging to say Dogs and Cats, when done a Dot product with a **trained weight vector**, will fall into either the Dog space of the hyperplane, or the Cat space of the hyperplane. This is how neural networks do classifications.

*Concept of Hyperplane*

![hyperplane2]

Next let's see how Perceptron is trained next.

## Contents

- **Chapter 1: [The simplest Neural Network - Perceptron using Vectors and Dot Products](1_vectors_dot_product_and_perceptron.md)**
- Chapter 2: [Perceptron Training via Feature Vectors and Dot product ](2_perceptron_training.md) 
- Chapter 3: [Gradient Descent, Gradient Vector and Loss Function](3_gradient_descent.md)
- Chapter 4: [Activation functions, Cost functions and Back propagation](4_backpropogation.md)
- Chapter 5: [Implementing a Neural Network using Chain Rule and Back Propagation](5_backpropogation_matrix_calulus.md)
- Chapter 6: [A Simple NeuralNet with above Equations](6_neuralnetworkimpementation.md)


  [1]: https://en.wikipedia.org/wiki/Vector_space
  [2]: http://www.mathcentre.ac.uk/resources/uploaded/mc-web-mech1-5-2009.pdf
  [3]: https://i.stack.imgur.com/Q1rBUm.png#center
  [4]: https://i.stack.imgur.com/t0plRm.png#center
  [Euclidean_vector]: https://en.wikipedia.org/wiki/Euclidean_vector
  [6]: https://i.stack.imgur.com/kO3ym.png#center
  [7]: https://i.stack.imgur.com/Nw2Ls.png#center
  [8]: http://tutorial.math.lamar.edu/Classes/CalcII/DotProduct.aspx
  [9]: https://sergedesmedt.github.io/MathOfNeuralNetworks/VectorMath.html#learn_vector_math_diff
  [10]: https://alan.do/minskys-and-or-theorem-a-single-perceptron-s-limitations-490c63a02e9f
  [11]: https://maelfabien.github.io/deeplearning/Perceptron/#the-classic-model
  [12]: https://sergedesmedt.github.io/MathOfNeuralNetworks/RosenblattPerceptronArticle.html
  [13]: https://sergedesmedt.github.io/MathOfNeuralNetworks/RosenblattPerceptronArticle.html
  [hyperplane2]: https://i.imgur.com/9M8GZHc.png#center
  [hyperplane1]: https://i.imgur.com/OIN3maHm.png#center
  [indexnotation]: https://web.iitd.ac.in/~pmvs/courses/mcl702/notation.pdf
  [matrix decomposition]: https://en.wikipedia.org/wiki/Matrix_decomposition
