# The Maths behind Neural Networks

Alex Punnen \
&copy; All Rights Reserved \
2019-2021 

---

## Contents

- Chapter 1: [Vectors, Dot Products and  the Perceptron](1_vectors_dot_product_and_perceptron.md)
- Chapter 2: [Feature Vectors, Dot products and Perceptron Training](2_perceptron_training.md)
- Chapter 3: [Gradient Descent, Gradient Vector and Loss Function](3_gradient_descent.md)
- Chapter 4: [Activation functions, Cost functions and Back propagation](4_backpropogation.md)
- Chapter 5: [Back Propagation with Matrix Calulus](5_backpropogation_matrix_calulus.md)
- Chapter 6: [A Simple NeuralNet with above Equations](6_neuralnetworkimpementation.md)


### Chapter 1
## Vectors, Dot Products and  the Perceptron

To understand the maths or modelling of the neural network, it is best to start from the beginning when things were simple. The earliest neural network - the `Rosenblatt’s Perceptron` was the first to introduce the concept of using vectors and the property of dot product to split hyperplanes of input feature vectors. These are the fundamentals that are still used today. There is a lot of terms above, and the rest of the article is basically to illustrate these concepts from the very basics.

Before we talk about why Neural Network inputs and weights are modelled as vectors (and represented as matrices) let us first see what these mathematical concepts mean geometrically. This will help us in understanding the intuition of these when they are used in other contexts/ in higher dimensions.

## Vectors

A vector is an object that has both a magnitude and a direction. Example Force and Velocity. Both have magnitude as well as direction.

However we need to sepcify also a context where this vector lives -[Vector Space][1]. For example when we are thinking about something like [Force vector][2], the context is usually 2D or 3D Euclidean world.

![vector2D][3]

![vector3D][4]

(Source: 3Blue1Brown)

The easiest way to understand the Vector is in a geometric context, say 2D or 3D cartesian coordinates, and then extrapolate it for other Vector spaces which we encounter but cannot really imgagine. This is what we will try to do here.

## Matrices and Vectors

 Vectors are represented as matrices.A matrix is defined to be a rectangular array of numbers. Example here is a [Euclidean Vector][5] in three-dimensional Euclidean space (or $R^{3}$). So a vector is represented as a column matrix.

$$
a = \begin{bmatrix}
a_{1}\\a_{2}\\a_{3}\ 
\end{bmatrix} = \begin{bmatrix} a_{1} & a_{2} &a_{3}\end{bmatrix}
$$

## Dot product

Algebraically, the dot product is the sum of the products of the corresponding entries of the two sequences of numbers.

if $\vec a = \left\langle {a_1,a_2,a_3} \right\rangle$ and $\vec b = \left\langle {b_1,b_2,b_3} \right\rangle$, then $\vec a \cdot \vec b = {a_1}{b_1} + {a_2}{b_2} + {a_3}{b_3}$

**Geometrically**, it is the product of the Euclidean magnitudes of the two vectors and the cosine of the angle between them

$$
 \vec a \cdot \vec b = \left\| {\vec a} \right\|\,\,\left\| {\vec b} \right\|\cos \theta 
$$

 ![dotproduct][6]

These definitions are equivalent when using Cartesian coordinates.
Here is a simple proof that follows from trigonometry [8] and [9]

## Dot Product and Vector Alignment

**If two vectors are in the same direction the dot product is positive and if they are in the opposite direction the dot product is negative.** This can be visualized geometrically putting in the value of the Cosine angle.

So we could use the dot product as a way to find out if two vectors are aligned or not.

Imagine we have a problem of  classifying if a leaf is healthy or not based on certain features of the leaf. For each leaf we have some feature vector set.

 For any  **input feature vector** in that vector space, if we have a **weight vector**, whose dot product with one feature vector of the set of input vectors of a certain class (say leaf is healthy) is positive and with the other set is negative, then that weight vector is splitting the feature vector hyper-plane into two.
 
 **In essence, we are using the weight vectors to split the hyper-plane into two distinctive sets.**

 So any new leaf, if we only extarct the same features into a feature vector; we can dot it with the *trained* weight vector and find out if it falls in healthy or deceased class.

 Not all problems have their feature set which is linearly seperable. So this is a constraint of this system.

## Perceptron. The first Artificial Neuron

The initial neural network - the **Frank Rosenblatt's perceptron** was doing this and could only do this - that is finding a solution if and only if the input set was linearly separable.

## And the first AI Winter

 Note that the fact that Perceptron could not be trained for XOR or XNOR; which was demonstrated in 1969, by by Marvin Minsky and Seymour Papert in a famous paper,that showed that it was impossible for these classes of network to learn such non seperable feature sets.
 
 This led to the first *AI winter*, as much of the hype generated intially by Frank Rosenblatt's discovery became a disillusionment.

 ![linearseperable]
 

## Modelling of the Perceptron
Here is how the Rosenblatt's perceptron is modelled

  ![perceptron2][7]

[Image source ][11]

Inputs are $x_1$ to $x_n$ , weights are some values that are learned $w_1$ to $w_n$. There is also a bias (b)  which in above is  -$\theta$

The bias can be modelled as a a weight $w_0$ connected to a dummy input $x_0$ set to 1.

If we ignore bias for a second, the output $y$ can be written as the sum of all inputs times the weights thresholded by the sum value being greater than zero or not.

$$
y = 1  \text{ if } \sum_i w_i x_i \ge 0 \text{  else } y=0
$$

The big blue circle is the primitive brain of the primitive neural network - the perceptron brain. Which is basically a function $\sigma$ (sigma).

 This is what is called as an **Activation Function** in Neural Networks. We will see that later. This is a step function we use here, output is non continuous (and hence non-differentiable) and is either 1 or 0.

If the inputs are arranged as a column matrix and weights also arranged likewise then both the input and weights can be treated as vector and $\sum_i w_i x_i$ is same as the dot product $\mathbf{w}\cdot\mathbf{x}$. Hence the activation function can also be written as 

$$
\sigma (x) =
\begin{cases}
1, & \text{if}\ \mathbf{w}\cdot\mathbf{x}+b \ge 0 \\
0, & \text{otherwise} \\
\end{cases}
$$

Note that dot product of two matrices (representing vectors), can be written as that transpose of one multiplied by another, $w \cdot x = w^Tx$ 

$$
\sigma(w^Tx + b)=
\begin{cases}
1, & \text{if}\ w^Tx + b \ge 0 \\
0, & \text{otherwise} \\
\end{cases}
$$

All three equations are the same.

### The equation $w \cdot x \gt b$  defines all the points on one side of the hyperplane, and $w \cdot x \ge b$  all the points on the other side of  the hyperplane and on the hyperplane itself.

### This happens to be the  very definition of “linear separability” 

### **Thus, the perceptron allows us to separate our feature space in two convex half-spaces**
Ref ([12])

If we can calculate/train/learn the weights, then we can have a weight vector, which splits the input feature vectors to two regions by a hyperplane. This is the essence of the Perceptron, the intial artificial neuron.

![hyperplane1]

[Image source][13]

In simple terms, it means that an unknown feature vector of an input set belonging to say Dogs and Cats, when done a Dot product with a trained weight vector, will fall into either the Dog space of the hyperplane, or the Cat space of the hyperplane. This is how neural networks do classifications.

*Concept of Hyperplane*

![hyperplane2]

## Contents

- **Chapter 1: [Vectors, Dot Products and  the Perceptron](1_vectors_dot_product_and_perceptron.md)**
- Chapter 2: [Feature Vectors, Dot products and Perceptron Training](2_perceptron_training.md) 
- Chapter 3: [Gradient Descent, Gradient Vector and Loss Function](3_gradient_descent.md)
- Chapter 4: [Activation functions, Cost functions and Back propagation](4_backpropogation.md)
- Chapter 5: [Implementing a Neural Network using Chain Rule and Back Propagation](5_backpropogation_matrix_calulus.md)
- Chapter 6: [A Simple NeuralNet with above Equations](6_neuralnetworkimpementation.md)


  [1]: https://en.wikipedia.org/wiki/Vector_space
  [2]: http://www.mathcentre.ac.uk/resources/uploaded/mc-web-mech1-5-2009.pdf
  [3]: https://i.stack.imgur.com/Q1rBUm.png#center
  [4]: https://i.stack.imgur.com/t0plRm.png#center
  [5]: https://en.wikipedia.org/wiki/Euclidean_vector
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
  [linearseperable]: https://i.imgur.com/jmWvoWh.png