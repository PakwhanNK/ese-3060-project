# Part 1 Report

## Experiment 1: Change SGD to Muon Optimizer

First, we consider changing the optimizer. Choosing the right optimizer can greatly reduce training speed by making it faster to converge to the local optimum value. In the original file, we use a standard SGD with momentum of 0.85, which is relatively high. SGD is simple to compute but it can take many many iterations to converge. The Muon optimizer (Jordan, 2024) adds an orthogonalization step to every weight update. Standard SGD can cause weights to move in the direction of the gradient, but the weight matrices tend to stretch information more in some directions than others, resulting in instability that causes the model to slow down in order to deal with these distortions. Transformers are particularly sensitive to this distortion since the forward and attention layers all involve multiplications of large matrices.

By using a Muon optimizer, we take the momentum update and project it on a space of near-orthogonal matrices using the Newton-Schulz iteration. By doing so, we hypothesize that it will allow CIFAR-10 to train with more aggressive updates and improve stability.

You can find the experimental logs under part1exp1.json.

[Results and statistical analysis to be inserted here.]
