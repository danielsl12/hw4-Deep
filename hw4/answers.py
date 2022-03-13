r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""
import torch


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(
        batch_size=32, gamma=0.99, beta=0.5, learn_rate=1e-3, eps=1e-8, num_workers=2,
    )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    # hp["in_features"] = 8
    # hp["out_actions"] = 4
    hp["width"] = 0
    hp["height"] = 512
    hp["beta"] = 0.5
    hp["batch_size"] = 16
    hp["learn_rate"] = 7.5e-4

    # hp['hidden_layers'] = [128, 128]
    # hp["batch_size"] = 8
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(
        batch_size=32,
        gamma=0.99,
        beta=1.0,
        delta=1.0,
        learn_rate=1e-3,
        eps=1e-8,
        num_workers=2,
    )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp["width"] = 1
    hp["height"] = 196
    hp["beta"] = 0.5
    hp["delta"] = 0.5
    hp["batch_size"] = 16
    hp["learn_rate"] = 2e-4
    # hp["learn_rate"] = 0.00133742069
    # ========================
    return hp


part1_q1 = r"""
Subtracting a baseline in the Policy gradient decreases the size of the weights multiplied by the gradients
of the log-prob of the Policy. $\\$
As we understand, a baseline is a function that when added to an expectation, does not change the expected value,
but at the same time, it can significantly affect the variance.
In general, this is helpful when the scales of the reward for different trajectories changes dramatically causing large variance for the gradients. $\\$
Let's look at this example: $\nabla_{\theta} log \pi_{\theta}(\tau) = [0.8, 0.15, 0.05]$ for three different trajectories.
Furthermore the rewards are $r(\tau)$ = [500, 50, 20] (respectively).

We get that $ Var( [0.8 \cdot 500, 0.15 \cdot 50, 0.05 \cdot 20] ) ~= 52216 $

Let's take the mean reward 190 as a baseline and recalculate.
$ Var( [0.8 \cdot (500-190), 0.15 \cdot (50-190), 0.05 \cdot (20-190)] ) ~= 23051 $
"""

part1_q2 = r"""
We know that $ v_{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \cdot q_{\pi}(s,a) $
$\\$
Because we take the mean over many trajectories, we hope that due to the law
of large numbers our optimization will push
$\hat{v}_{\pi}(s)$ to be approximately $\mathbb{E}({\hat{q}_{i,t}|s_0 = s,\pi})$
(we assume that the many trajectories implicitly factor in the 
probability of each action given a state). $\\$
Since:
$
q-values= \hat{q}_{i,t} = \sum_{t'\geq t} \gamma^{t'}r_{i,t'+1}
$
$
g_t(\tau) = r_{t+1}+\gamma r_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+1+k}
$

This is more or less $\mathbb{E}({g(\tau)|s_0 = s,\pi})$ which is the definition of $v_{\pi}(s)$.

This is why choosing the estimated q-values as regression targets for our state-values leads us to a valid
approximation of the value function.
"""

part1_q3 = r"""
1. Let's start by looking at the policy loss graphs. We can see that the non-baseline experiences started from
a low score and throughout training improved. For the baseline experiences however, we see that both are constant at
around 0. This happens because the baseline 
term we subtract from the weight of each sample normalizes the loss to be around zero, while
only the baseline changes through the training as we see in the baseline graph. 
We can see that for the baseline methods, the baseline, which represents the average q value
raises which means the model improves. $\\$
By examining the entropy loss graph we can see that the nagative loss rises, which means that the entropy drops. This 
is happening until some point in training which after that point, the loss remains at around the same area of values. Our 
guess as to why this happens is because the policy loss takes more weight of the total loss. (i.e. the policy loss affects 
the total loss more than the entropy loss), as a result, the model chooses better policies instead of more diverse ones.
$\\$ Finally, by looking at the mean-reward graph, we can see that for all 4 experiences, the reward goes up. Based on 
that we can understand that throughout the training process, our model does improve and does learn how to "play the game" 
better. $\\$
2. Let's start with examining the policy loss graph. As mentioned above, the CPG experience is stable at around 0. As for 
the AAC, we can see that it starts at around the same area as the other non-baseline experiences, but at some point got a loss 
higher than 0 and from that point onward tried to stabilize at around this value. AAC doesn't appear in the baseline graph 
(as expected), and as for the entropy loss we can notice a similar result to the policy loss. The loss of the AAC 
experience starts at around the same area as the others, but at some point in the training, passes their values. $\\$
As for the mean-reward graph, we can clearly see that AAC got better results than the other experiences, which of course 
includes CPG. One more thing we can notice is that at the beginning of the training process, it was CPG that got better 
results however, as time progressed, AAC continued to improve with at a larger pace and surpassed the results of CPG.
"""

# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    # descent results:
    hypers = dict(
        batch_size=16, h_dim=1024, z_dim=16, x_sigma2=0.001, learn_rate=0.0002, betas=(0.9, 0.99),
    )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=16,
        z_dim=128,
        data_label=1,
        label_noise=0.2,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            # weight_decay=0.001,
            betas=(0.5, 0.999)
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas=(0.5, 0.999),
            # weight_decay=0.002
            # weight_decay=0.002
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    # hypers = dict(
    #     batch_size=9,
    #     z_dim=75,
    #     data_label=1,
    #     label_noise=0.2,
    #     discriminator_optimizer=dict(
    #         type="Adam",  # Any name in nn.optim like SGD, Adam
    #         lr=0.0001,
    #         # You an add extra args for the optimizer here
    #         betas=(0.6, 0.99)
    #     ),
    #     generator_optimizer=dict(
    #         type="Adam",  # Any name in nn.optim like SGD, Adam
    #         lr=0.001,
    #         # You an add extra args for the optimizer here
    #         betas=(0.6, 0.99)
    #     ),
    # )
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============


# ==============
# Part 4 answers
# ==============


def part4_affine_backward(ctx, grad_output):
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
