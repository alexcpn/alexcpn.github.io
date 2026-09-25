<!-- Replace the single backprop_1 image in Chapter 4 (Quarto markdown) with this 2x2 figure -->
::: {layout-ncol=2}
![**(a) Nudge the weight.** Make a small change $\Delta w$ to the weight $w^{(l)}$.](images/bp_step1.png)

![**(b) It changes the weighted input.** $z^{(l)}$ shifts by $\Delta z$; the rate is $\frac{\partial z^{(l)}}{\partial w^{(l)}} = a^{(l-1)}$.](images/bp_step2.png)

![**(c) It changes the activation.** $a^{(l)}$ shifts by $\Delta a$; the rate is $\frac{\partial a^{(l)}}{\partial z^{(l)}} = \sigma'(z^{(l)})$.](images/bp_step3.png)

![**(d) It changes the cost.** $C_0$ shifts by $\Delta C_0$. Multiplying the three local rates gives $\frac{\partial C_0}{\partial w^{(l)}}$: the chain rule.](images/bp_step4.png)
:::
