
# Backpropogation

## Activation functions, Cost functions and Backpropogation

We have seen how gradient descent was used to optimise the cost function associated with linear regression and how that leads to find the optimal line seperating the features  in  linear regression.

Now this same method is used for Neural Network Learning. We use the method of backpropogation to propogate recursively the optmised weights from gradient descent from output layers to the input layers.

## Structure of a Neural Network

Neural network is basically a set of inputs connected through 'weights' to a set of activation functions - the artifical neuron (the round circles you see in the diagram below), whose output will be the input for the next layers and so on. The final layer will be either 2 'neurons' for modelling classification problems, or a set of connections for regression problems - example for number recogintion the output will have 10 neurons (to represent digits 0 to 9). Below is one diagram.

![neuralnetwork]

For training a neural network we need a dataset which has the input and expected output. The weights are randomly intialised and the inputs passed into the activation function through the weights gives some ouput. This output or computed values can be compared to the expected results and the difference gives the error of the network. We can create a 'Cost Function' with a simple function of this error like what we saw in the last chapter say the Mean Squared Error. We can use the same Gradient Descent now to adjust the weights so that the error is minimized. 

So far so good.

 We do this in the last layer of the multi-layer neural network. 
 
 Now how do we adjust the weights of the inner layer ? This is where Backpropogation comes in. In a recursive way weights are adjusted as per their contributiion to the output. That is weights or connections which have influenced the output more are adjusted more than those which have influenced the output less.

 ## How Backpropogation works

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

So bacically neural network is a chain of activations, one feeding into another.

Let the explected output be $y$ for a training example $x$. Notice that we train with a lot of examples. But we are talking here now about only one example in the training set.

The calculated output for a network with $l$ layer is $a^l$.

Let's calculate then the loss function. Difference of expected to the calculated.

Let's use the quadratic cost function here. 


$$
 C = \frac{1}{2} \|y-a^L\|^2 
$$


---
Note 

If there are  $j$ output layers, we need to take the sum of all the activations of the $j$ layers. But this complicates the notation and obscures the intution. So lets skip this for now

$$
C = \frac{1}{2} \sum_j (y_j-a^L_j)^2,
$$
---

Now this Cost needs to be reduced. We can use the gradient descent to find the weight that reduces the cost function for a set of training examples.

Note that this is the weight of the last layer (layer l). What backpropogation does is to adjust the weights of the other layers according to how strong was their influence on its forward layers.

We have two layers here. We have got the weight of layer $l$. We need to find how much to adjust the previous layer $l-1$.

The best way to see this visually is the way it is done by the tree representation of 3Blue1Brown video.

Here is the crux of this below. This is a representaion of a single path in the last layer (l); and it shows how the connection from previous layer; that is the activation of the previous layer and the weigh of the current layer is affecting the output; and thereby the final Cost.

The central idea is how a change in weight affects the Cost in this chain depiction.

$$
\delta C_0/\delta w^l = \delta z^l/\delta w^l . \delta a^l/\delta z^l . \delta C_0/\delta a^l 

$$

We need to find how a small change in weight changes the cost. This is equal to the change in $z^l$ due to change in $w^l$, and change in $a^l$ due to change in $z^l$ and change in $C_0$ by chante in $a^l$. This is the chain rule and this graph representation explains this very intutively.


![backpropogationgif]

Here is a more detailed depiction of how the small change in weight add's through the chain to affect the final cost.

![backpropogationgif2]

Now we have the chain rule, we can calculate how a small change in weight is going to change the Cost. This is for a single link. Like this there are lot of links from one layer to the next. 

This calculated number signifies how much a small nudge in the weight of a connection changes the output weight.

Basically this is the **gradient** of the Loss function or Cost function with respect to the weight of the network for a single input output example


Neurons that fire togehter, wire together. This is the sort of adage that is going on behind here. Basically the links that are contributing more to the cost change are adjusted more compared to those that are contributing less.

We now adjust the weights in each layer in propotion to how each layers weight affects the Cost function. (the propotion is what we calcualted by chain rule)

$$

  W^l_{new} = W^l_{old} - learningRate* \delta C_0/ \delta w^l

$$

And that's it folks, backpropogation demystified.


 



[neuralnetwork]: https://i.imgur.com/gE3QKCf.png
[backpropogation]: https://i.imgur.com/1s89fsX.png
[backpropogationgif]: https://i.imgur.com/jQOLUG3.gif
[backpropogationgif2]: https://i.imgur.com/AgyuOr2.gif







