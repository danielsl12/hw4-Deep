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
        batch_size=16, h_dim=1024, z_dim=16, x_sigma2=0.001, learn_rate=0.0003, betas=(0.9, 0.99),
    )
    # hypers = dict(
    #     batch_size=64, h_dim=1024, z_dim=200, x_sigma2=0.02, learn_rate=0.0003, betas=(0.5, 0.99),
    # )
    # ========================
    return hypers


part2_q1 = r"""
The $\sigma^2$ parameter affects the variance of the distribution over the latent space. It 
determines the relative strength of each of the terms in the loss function. A low value 
of $\sigma^2$ will result in the images generated from the model being more similar to each other, and 
a high value will result in the images being more distinct from one another. $\\$ 
"""

part2_q2 = r"""
1. The reconstruction loss represents the difference between the regenerated image and the original one. $\\$
The KL divergence loss can be interpreted as a regularization term. It is an upper bound to $-\mathop{\mathbb{E}}_x(log p(X))$, 
which we try to bring to a minimum in order to maximize $p(X)$. $\\$
2. The KL divergence minimizes the difference between the latent space and the space from which we sample. 
Since we sample from normal distribution, the affect of the loss on the latent space is making it more similar (or closer) 
to the normal distribution. $\\$
3. The affect of the KL divergence helps us quite a lot. By making the latent space more similar to the normal distribution, 
we allow our model to sample $z$'s that are similar to the encoded vector (although not identical). This vector will 
make the decoder learn more similar vectors to the original input, and generate better vectors which are closer to the 
desired result. $\\$
"""

part2_q3 = r"""
The reason for this is the lower bound of the KL divergence term. As we saw, the lower bound we get is: 
$log p(X) \geq  \mathop{\mathbb{E}}_{z\sim q_{α}} (log p_{β}(X|z))-D_{KL}(q_{α}(Z|X)||p(Z))$ . Because $p(X)$ is the 
probability of a given instance $X$ under the entire generative process, and we aim to maximize this probability for each 
instance. This can also be thought of as minimizing the loss $-\mathop{\mathbb{E}}_x(log p(X))$ which is intractable. $\\$
"""

part2_q4 = r"""
The reason is to ensure numerical stability. The standard values are mostly small numbers between 0 to 1. Working with 
those low values may lead us to errors because really low numbers are often viewed by the computer as zeros. By instead 
using the log we are able to shift the domain we work on from $[0,1]$ to $[-∞,0]$ , allowing a much bigger space for 
operations and numerical stability. $\\$
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=16,
        z_dim=128,
        data_label=1,
        label_noise=0.2,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas=(0.5, 0.999)
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas=(0.5, 0.999),
            # You an add extra args for the optimizer here
        ),
    )    # ========================
    return hypers



part3_q1 = r"""
We maintain the gradients when we train the generator and we discard the gradients when we train the discriminator.
We do it because when we train the discriminator we don't want to backpropagate through the generator because it will lead
to the generator learning to fool the discriminator (the generator learns how to minimize the discriminator loss)
rather than learning to create good images. 
"""

part3_q2 = r"""
 1. No, because when we decide stop training only based on the generator loss it might be that the generator loss is low 
 because it is fooling the discriminator but the discriminator loss is high which means that the generator not producing  
 quality images. Basically, the generator loss is low not because he is creating good images but because the discriminator 
 can't distinguish between real images to fakes ones.
 
 2. It means that the generator is able to fool the discriminator and because the discriminator loss is constant 
 it also means that the generator is still learning and improving. 
"""

part3_q3 = r"""
As we look at the results we got from the two models we can see that the main difference between the models is that the 
images that are being generated by the GAN are sharper but with a lot of artifacts and the background is less coherent, 
while the images generated by the VAE model are more blurry, but the background looks more real but as we said blurry. 
The main reason for this is the way the 2 models calculate their loss. $\\$ 
In the GAN model, the loss of the generator is evaluated by the amount of images that fooled the discriminator. 
As a result, the generator tries to create image with specific features (such as facial features like eyes, 
mouth and nose) and ignores most of the background. This is why we get images with sharp faces but blurry background. $\\$ 
On the other hand, the VAE model tries to generate images to be as close as they can be to the original images. As 
explained in the previous part, the model minimizes the distance between the generated image and the original, which 
creates blurry images because of the "Regression to the mean" problem. $\\$ 
"""

# ==============


# ==============
# Part 4 answers
# ==============


def part4_affine_backward(ctx, grad_output):
    # ====== YOUR CODE: ======
    X, W, b = ctx.saved_tensors
    dx = torch.matmul(0.5 * W.T, grad_output.T)
    dw = torch.matmul(0.5 * grad_output.T, X)
    db = grad_output

    return dx, dw, db
    # ========================
