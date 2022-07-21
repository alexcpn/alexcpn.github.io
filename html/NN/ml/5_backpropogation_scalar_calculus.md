# The Maths behind Neural Networks

Alex Punnen \
&copy; All Rights Reserved \

---

[Contents](index.md)

# Chapter 5

## Back propagation - Pass 2 (Scalar Calculus)


Let's consider the following two layer neural network

```python
x is the Input
y is the Output.
l is the number of layers of the Neural Network.
a is the activation function ,(we use sigmoid here)
```

$$
 x \rightarrow a^{l-1} \rightarrow  a^{l} \rightarrow  y
$$

Where the activation $a^l$ is sigmoid here
$$
  a^{l} = \sigma(w^l a^{l-1}+b^l).
$$

We will try to derive out via back propogation the effect of the weights in an inner layer to the final loss.

There is one caveat here;these equations are just illustrative with respect to scalar calculus and not  accounting for the matrix calculus we will need when modelling a practical neural network. The maths is a bit complex with matrix calculus and we can start with the simplified discourse with Chain rule.

We can write $a^l$ as

$$
a^{l} = \sigma(z^l) \quad where \quad

z^l =w^l a^{l-1} +b^l
$$

We can also easily calculate

$$
\mathbf {\frac{\partial a^{l} }{\partial z^l} = \frac{\partial \sigma (z^{l}) }{\partial z^l} = \sigma' (a^{l}) \quad \rightarrow  ( {a})}
$$

Where $\sigma'$ = derivative of Sigmoid with respect to Weight

Note $x$ can also be written as $a^0$

---

Our two layer neural network can be written as

 $$
 \mathbf { a^0 \rightarrow a^{1} \rightarrow  a^{2} \rightarrow  y }
 $$

 ---

 Note that $a^2$ does not denote the exponent but just that it is of layer 2.

Lets write down the Chain rule first.

$$
\mathbf {
\frac {\partial C}{\partial w^l} = \frac {\partial z^l}{\partial w^l} . \frac {\partial a^l}{\partial z^l} . \frac {\partial C}{\partial a^l}
= \frac {\partial a^l}{\partial w^l} . \frac {\partial C}{\partial a^l}
}
$$

We will use the above equation as the basis for the rest of the chapter.

## Gradient Vector of Loss function In Output Layer

Lets substitute $l$ and get the gradient of the Cost with respect to weights in layer 2 and layer 1.

### For the last layer - Layer 2

$$
\mathbf {
\frac {\partial C}{\partial w^2} = \frac {\partial z^2}{\partial w^2} . \frac {\partial a^2}{\partial z^2} . \frac {\partial C}{\partial a^2}
}
$$

The first term is

$$
\mathbb{
\frac{\partial z^{2} }{\partial w^2} = \frac{\partial (a^1.w^2)}{\partial w^2} =a^1 \quad \rightarrow  (\mathbf  {1.1})
}
$$

The second term is

$$
\mathbb{
\frac{\partial a^{2} }{\partial z^2} = \frac{\partial \sigma(z^2) }{\partial z^2} =\sigma' (z^{2}) \quad \rightarrow  (\mathbf  {1.2})
}
$$

The third term is

$$
\mathbf{
\frac{\partial C}{\partial(a^2)} = \frac {\partial({\frac{1}{2} \|y-a^2\|^2)}}{\partial(a^2)} = \frac{1}{2}*2*(a^2-y) =(a^2-y) \rightarrow (1.3) }
$$

Putting 1.1,2.1 & 3.1  together we get the final equation for the second layer. This is the output layer.

---

$$ \mathbf{
\frac {\partial C}{\partial w^2} =  a^1* \sigma' (z^{2})*(a^2-y) \quad \rightarrow (A) }
$$

---

## Gradient Vector of Loss function in Inner Layer

Now let's do the same for the inner layer.

$$

\frac {\partial C}{\partial w^1}= \frac {\partial z^1}{\partial w^1}. \frac {\partial a^1}{\partial z^1}. \frac {\partial C}{\partial a^1}
$$

The first term is  similar to (1.1)
$$
\mathbb{
\frac{\partial z^{1} }{\partial w^1} = \frac{\partial a^0.w^1}{\partial w^1} =a^0 \quad \rightarrow  (\mathbf  {2.1})
}
$$

The second term is also similar to (1.2)

$$
\mathbb{
\frac{\partial a^{1} }{\partial z^1} = \frac{\partial \sigma(z^1) }{\partial z^1} =\sigma' (z^{1}) \quad \rightarrow  (\mathbf  {2.2})
}
$$

For the third part, we use Chain Rule to split like below, the first part of which we calculated in the earlier step. This is where Chain Rule helps.

$$
\frac{\partial C}{\partial(a^1)} =  \frac{\partial C}{\partial(a^2)}.\frac{\partial(a^2)}{\partial(a^1)}
$$

$$\begin{aligned}

Note \space that \space in \space the\space  previous \space section \space \space  we \space had \space calculated \quad

\frac {\partial C}{\partial(a^2)}  =(a^2-y)  \rightarrow (2.3.1)\\ \\

Now \space to \space calculate \quad

 \frac{\partial(a^2)}{\partial(a^1)} \space  \\ \\

We \space can \space re-write  \space this \space as \\ \\

 \frac{\partial(a^2)}{\partial(a^1)} =  \frac{\partial(a^2)}{\partial(z^2)}. \frac{\partial(z2)}{\partial(a^1)}   \\ \\

 which \space is \space \\ \\ 

  \frac{\partial \sigma (z^2)}{\partial(z^2)} .\frac{\partial(w^2.a^1)}{\partial(a^1)} \\ \\

 which \space is \space \\ \\ 

 \sigma'(z^2).w^2 \\ \\

\frac{\partial(a^2)}{\partial(a^1)} = \sigma'(z^2).w^2  \quad \rightarrow (2.3.2)\\ \\

\end{aligned}$$

Putting  (2.1),(2.2),(2.3.1)and (2.3.2)  together, we get

---

$$
\mathbf{
\frac {\partial C}{\partial w^1} =a^0* \sigma'(z^1)*(a^2-y).\sigma'(z^2).w^2 \quad \rightarrow \mathbb (B)
}
$$

---

Repeating here the previous equation (A) as well

$$ \mathbf{
\frac {\partial C}{\partial w^2} =  a^1* \sigma' (z^{2})*(a^2-y) \quad \rightarrow (A) }
$$

---

- Next [A Simple NeuralNet with  Back Propagation](6_neuralnetworkimpementation.md)
