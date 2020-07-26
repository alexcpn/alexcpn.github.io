
# Back-propagation Demystified

## Activation functions, Cost functions and Back-propagation

We have seen how gradient descent was used to optimise the cost function associated with linear regression and how that leads to find the optimal line separating the features  in  linear regression.

Now this same method is used for Neural Network Learning. We use the method of back-propagation to propagate recursively the optimise weights from gradient descent from output layers to the input layers.

## Structure of a Neural Network

Neural network is basically a set of inputs connected through 'weights' to a set of activation functions - the artificial neuron (the round circles you see in the diagram below), whose output will be the input for the next layers and so on. The final layer will be either 2 'neurons' for modelling classification problems, or a set of connections for regression problems - example for number recognition the output will have 10 neurons (to represent digits 0 to 9). Below is one diagram.

![neuralnetwork]

For training a neural network we need a dataset which has the input and expected output. The weights are randomly initialized and the inputs passed into the activation function through the weights gives some output. This output or computed values can be compared to the expected results and the difference gives the error of the network. We can create a 'Cost Function' with a simple function of this error like what we saw in the last chapter say the Mean Squared Error. We can use the same Gradient Descent now to adjust the weights so that the error is minimized.

So far so good.

 We do this in the last layer of the multi-layer neural network.

 Now how do we adjust the weights of the inner layer ? This is where Backpropagation comes in. In a recursive way weights are adjusted as per their contribution to the output. That is weights or connections which have influenced the output more are adjusted more than those which have influenced the output less.

## How Backpropagation works

 Let us see the mathematical notation of representing  a neural network.

 A real neural network has many layers and many interconnections. But let's first think of a simple two layered neural network with just a linear connection between two layers, l and (l-1) to simplify the notation and to understand the maths.

$$
 x \rightarrow a^{l-1} \rightarrow  a^{l} \rightarrow  output
 $$

   We take  $a^{l}$  as the input at layer *l*. Input  of a neuron in layer *l*  is the output of activation from the previous layer $(l-1)$.

 Output of Activation is  the product of the weight *w* and input at layer $(l-1)$  plus the *basis*, passed to the *activation function*. We will be using the sigmoid ($\sigma$ ) function as the activation function.

Writing this below gives.

$$
  a^{l} = \sigma(w^l a^{l-1}+b^l).
$$

http://neuralnetworksanddeeplearning.com/chap2.html

So basically neural network is a chain of activations, one feeding into another.

Let the expected output be $y$ for a training example $x$. Notice that we train with a lot of examples. But we are talking here now about only one example in the training set.

The calculated output for a network with $l$ layer is $a^l$.

Let's calculate then the loss function. Difference of expected to the calculated.

Let's use the quadratic cost function here.

$$
 C = \frac{1}{2} \|y-a^L\|^2
$$

---
Note

If there are  $j$ output layers, we need to take the sum of all the activations of the $j$ layers. But this complicates the notation and obscures the intuition. So lets skip this for now

$$
C = \frac{1}{2} \sum_j (y_j-a^L_j)^2,
$$

---

### Gradient Descent

Now this Cost needs to be reduced. We can use the **gradient descent** to find the path to the optimal weight that reduces the cost function for a set of training examples.

As we have seen earlier in gradient descent chapter, we get the path to the optimal weight by following the negative of the gradient of the Cost function with respect to the weight. But in multi-layered neural network  the question gets tricky on how to update the weights in the different layers.This is where Backpropagation algorithm comes in.

### Back-Propagation

There are various definitions. This is not easy to grasp, but by the end of this chapter, you should have a pretty good grasp.

The Wikipedia definition - The term backpropagation strictly refers only to the algorithm for computing the gradient - that is computing $\Delta C / \Delta w$, computing the gradient of the Loss or Cost function with respect to the weight.

 Speaking more generally, what we usually mean by Back Propagation is the mechanism to adjust the weights of all the layers according to how strong was *each*  of their influence on the final Cost.

 Speaking more specifically - It is an algorithm to adjust each weight of every layer in a neural network, by using gradient descent, by calculating the gradient of the Cost function in relation to each weight.

If you have not got this explanation fully, this is fine, once you understand it working, the above will be apparent naturally.

### Back-Propagation in Detail

Consider a neural network with multiple layers. The weight of layer $l$ is $w^l$.  And for the previous layer it is $w^{(l-1)}$.

The best way to understand backpropagation is visually and by the way it is done by the tree representation of 3Blue1Brown video linked [here](https://www.youtube.com/watch?v=tIeHLnjs5U8).

 The below  GIF is a representation of a single path in the last layer($l$ of a neural network; and it shows how the connection from previous layer - that is the activation of the previous layer and the weight of the current layer is affecting the output; and thereby the final Cost.

The central idea is how a change in weight affects the Cost in this chain depiction.

There are two three things to note. We are talking now about the *Cost* and how the Cost and Weights in different layers are related.


We need to find how a small change in weight ($\Delta w$), shown int the top left, changes the Cost. This change of Cost, is the result of the change in $z^l$ due to change in $w^l$, and change in $a^l$ due to change in $z^l$ and change in $C_0$ by change in $a^l$. This is the Chain Rule and this graph representation explains this very intuitively.

![backpropogationgif]

Here is a more detailed depiction of how the small change in weight adds through the chain to affect the final cost.

![backpropogationgif2]

This is the Chain Rule 

$$

\delta C_0/\delta w^l = \delta z^l/\delta w^l . \delta a^l/\delta z^l . \delta C_0/\delta a^l

$$


Now we have the Chain Rule, we can calculate how a small change in weight is going to change the Cost through a chain of layers

Ignore for the time that this is for a single link. In a production neural network there are lot of links from one layer to the next.

This calculated number $\delta C_0/\delta w^l$ signifies how much a small nudge in the weight of a connection changes the output weight.

This calculation is the key part. It is easy to grasp visually, once you have the connection diagram or *computational graph* of the network in mind.

Basically this is the **gradient** of the Loss function or Cost function with respect to the weight of the network for a single input output example. 

This is what BackPropagation calculates.Now the definition of Back Propagation may seem more understandable.

Next part of the recipe is adjusting the weights of each layers, depending on how they contribute to the Cost.

Neurons that fire together, wire together. This is the sort of adage that is going on behind here. Basically the links (which are the weights in a neural network) that are contributing more to the cost,are adjusted more, compared to those that are contributing less. Basically like strengthening the links that seem destined to be wired together, vaguely similar to how biological neurons wire together.

We now adjust the weights in each layer in proportion to how each layers weight affects the Cost function. (the proportion is what we calculated by chain rule - by Back Propagation)

This adjustment then is calculating the new weight by following the negative of the gradient of the Cost function - basically by gradient descent.

$$

  W^l_{new} = W^l_{old} - learningRate* \delta C_0/ \delta w^l

$$

For adjusting the weight in the  $(l-1)$ layer, we do similar

First calculate how the weight in this layer contributes to the final Cost or Loss

$$
\delta C_0/\delta w^{l-1} = \delta z^{l-1}/\delta w^{l-1} . \delta a^{l-1}/\delta z^{l-1} . \delta C_0/\delta a^{l-1}

$$

and using this

$$

  W^{l-1}_{new} = W^{l-1}_{old} - learningRate* \delta C_0/ \delta w^{l-1}

$$

And that's it folks, backpropagation demystified. Simple and elegant, but looks pretty complex from the outside.

Next would be to add more layers and more connections and change the notation to represent the place of each weight in each layer so $w^l$ becomes $w^l_{j,k}$

![weightnotation]
Source : Michael Nielsen: NeuralNetwork and Deep Learning book

Some other references.

http://neuralnetworksanddeeplearning.com/chap2.html

https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

[neuralnetwork]: https://i.imgur.com/gE3QKCf.png
[backpropogation]: https://i.imgur.com/1s89fsX.png
[backpropogationgif]: https://i.imgur.com/jQOLUG3.gif
[backpropogationgif2]: https://i.imgur.com/AgyuOr2.gif
[weightnotation]: https://i.imgur.com/XZT17pu.png
