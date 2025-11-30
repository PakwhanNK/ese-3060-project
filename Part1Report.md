# Part 1 Report

We came up with four ideas for systemic improvements to the NanoGPT architecture and processing pipeline. They involve changes to the normalization function, the QKV dimensionality representation, and the activation function.

## Experiment 1: RMSNorm Selective Skipping

This experiment tests whether we can skip certain RMSNorm operations during training when they are expected to have negligible effect. RMSNorm ensures stable activation magnitudes by normalizing each token vector, but prior work by Zhang et al. (2019) shows that RMSNorm is highly robust to small deviations in scale. Our hypothesis is that if a token’s vector norm is already close to 1.0, then RMSNorm would perform almost no adjustment, and thus can be safely skipped to reduce compute. Because RMSNorm appears twice per transformer block, this model performs 24 RMSNorm calls per forward pass in a 12-layer GPT-2—making it a substantial contributor to runtime. In this experiment, we introduce a conditional check on each RMSNorm call and bypass normalization when the deviation from unit norm is below a small threshold. We expect this to yield measurable speed improvements without harming final validation loss.

[Results, logs, and statistical analysis to be inserted here.]

## Experiment 2: Dynamic Tanh as a Replacement for Normalization

This is a completely different approach from Experiment 1. Instead of trying to maximize efficiency within RMSNorm computations, we consider replacing normalization with a learned activation function, drawing motivation from the “Transformers without Normalization” paper (Zhu et al., 2025). Instead of computing RMS statistics and loading learned gain parameters, we apply a lightweight activation: DyT(x) = tanh(αx), where α is a learned scalar per normalization site. The hypothesis is that this activation acts as a smooth, implicit normalization mechanism: tanh clips extreme activations to maintain bounded magnitude, while α allows the model to learn how sharply to scale or compress vector norms. Because DyT avoids both the RMS computation and memory loads associated with normalization, it should reduce forward-pass latency while preserving stability through the residual stream. All RMSNorm calls—in attention, in the MLP, and at the final layer—are replaced with DyT.

[Results, logs, and statistical analysis to be inserted here.]

## Experiment 3: Low-Rank QKV Projections

This experiment proposes reducing the dimensionality of the attention Q, K, and V projections to decrease the cost of attention without significantly harming predictive quality. Motivated by Linformer (Wang et al., 2020), which demonstrates that attention matrices are intrinsically low-rank, we hypothesize that the head dimension in NanoGPT can be safely reduced (e.g., from 128 → 96) via an additional learned projection layer. This reduces the width of Q, K, and V before computing attention, lowering the cost of QKᵀ, the softmax, and subsequent attention operations. Because the representational structure of attention has experimentally been found to possess excess dimensionality, we expect the model to tolerate such compression with minimal loss impact and measurable speed gains during training.

[Results, logs, and statistical analysis to be inserted here.]

## Experiment 4: Early-Exit via an Activation Gate

This experiment explores selectively skipping the MLP computation for tokens with very low activation strength. The feed-forward network represents the single most expensive operation in each transformer block, and many tokens arrive at the MLP with relatively small vector magnitudes. Since the MLP in NanoGPT uses ReLU² as its activation function, a very small input vector produces almost zero contribution after the activation. Following a paper called CALM (Schuster et al., 2022), which showed that token-wise conditional computation can reduce cost by 30–50% without harming accuracy, we introduce a gating rule: if the token’s vector norm falls below a threshold of 0.15, we return the input directly. Because this decision happens after attention, semantic structure and inter-token dependencies remain intact. This experiment evaluates whether such token-level skipping yields runtime reductions while preserving convergence quality.

[Results, logs, and statistical analysis to be inserted here.]
