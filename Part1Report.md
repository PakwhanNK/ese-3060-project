# Part 1 Report: Improving the CIFAR-10 Training Time
# 0. Relevant Links
- [1-page summary](https://docs.google.com/document/d/15eWZzH0Cznki9weSfbSCWqdPMxBw-PSbRJbYpdeu-4o/edit?usp=sharing) (PDF)
- [Experiment logs/screenshots](https://drive.google.com/file/d/1Xfst_8ZV70lJESHTTTa8IXQLbUAOQxnF/view?usp=sharing) (PDF)
- [Link to folder containing CSV results](https://drive.google.com/drive/folders/108H9awMt_TQRDSk0beFI4ZJINIC0P69T?usp=sharing)

# 1. Hypothesis

Building on the training methods in 94% on CIFAR-10 in 3.29 Seconds on a Single GPU, we group potential speedups into two categories: (1) direct methods that reduce the time needed to reach a target accuracy (e.g., changing learning-rate schedules or optimizer behavior), and (2) indirect methods that increase accuracy at fixed compute, so that we can later reduce the number of epochs while still exceeding 94% test accuracy.

Our experiments are therefore designed to test whether a change (a) decreases time to reach 94% accuracy or (b) increases accuracy at fixed training time. If a modification yields a statistically significant reduction in wall-clock time without lowering median accuracy below 94%, we consider merging it into the main branch. If, instead, it significantly boosts accuracy at a fixed time, we plan to keep the change and then reduce the number of epochs to trade excess accuracy for faster runs, consistent with the project goal.

**We formalize this into four specific hypotheses:**

1. **Changing the starting and ending values of the triangular learning rates will reduce the time required to achieve 94% accuracy.**
Intuitively, starting closer to the maximum learning rate or ending at a higher final rate may allow the optimizer to move quickly through “easy” regions of the loss landscape, shortening training. However, this may also reduce the granularity with which we explore learning rates and hurt final accuracy. Our hypothesis is that, in certain settings, the speed benefit outweighs the potential loss in accuracy.

2. **Changing the lookahead optimizer parameter update frequency and interpolation parameter can either reduce time-to-94% or increase final accuracy.**
Because Lookahead maintains fast and slow weights, updating the slow weights more frequently may help the optimizer escape small fluctuations and converge to a good solution in fewer effective steps. We hypothesize that a configuration exists where this improved stability translates into faster convergence or higher accuracy for the same number of epochs. 

3. **Changing the ratio of augmented to unaugment images in TTA (while keeping TTA level 2 fixed) can increase test accuracy without affecting training time.**
Since TTA is applied only at evaluation, adjusting the mixture of translated and untranslated views should not change the wall-clock training cost; however, an improved combination may yield higher accuracy. If the gain is large enough, we can later reduce the number of training epochs while maintaining a value above 94%.

4. **Decreasing the number of training epochs in small increments will reduce total training time while maintaining an accuracy of at least 94%.**
Because the baseline setup already reaches 94% with some margin, we expect a range of shorter schedules that maintain the target accuracy with proportionally less wall-clock time.

# 2. Methodology

## 2.1. Instrumentation and Code Improvements

Before running any experiments, we modified the project code to support reproducible, statistically sound comparisons across runs:

- ExperimentLogger class: Added automatic CSV logging of per-run statistics, including mean, standard deviation, and 95% confidence intervals for accuracy and wall-clock time, as well as “successful run” time (runs with accuracy ≥94%) and success rate.
- GPU metadata tracking: Logged nvidia-smi info, PyTorch/CUDA versions, RunPod instance details for reproducibility
- Git branch workflow: Each experiment series in a separate branch (experiment/exp2xx-lr-sweeps, experiment/exp3xx-lookahead, etc.) before merging only statistically supported improvements into main at the end.
- Modular triangle function: Refactored LR schedule into a reusable triangle(steps, start, end, peak) function with flexible parameterization

All experiments used the same GPU type, number of GPUs, dataset, and model configuration.

## 2.2 Baseline Experiment

Because runtime depends on the specific laptop / GPU environment, we could not directly use the paper’s reported 3.29s as our baseline. Instead, we created a branch experiment (exp000-baseline-100-runs) and ran 100 baseline training sessions to estimate our own mean time to 94% accuracy and its variance. This baseline serves as the control condition for all subsequent modifications.

## 2.3. Hyperparameter Sweeps

To explore speed–accuracy tradeoffs, we ran a series of hyperparameter sweeps with 25 epochs per configuration. The goal at this stage was to map how each parameter affects accuracy and runtime, rather than fully optimizing the number of epochs.

### 2.3.1. Triangular Learning-Rate Sweeps

All LR sweeps were implemented by modifying the arguments to the shared triangle() function.

1. Peak position sweep
	1. Branch: experiment/exp202-lr-peak
	2. Configurations: 8 peak positions in [0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36]
	3. Runs: 25 seeds per configuration (200 runs total)  
2. End value sweep
	1. Branch: experiment/exp201-lr-end
	2. Configurations: 7 end values in [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
	3. Runs: 25 seeds per configuration (175 runs total)  
3. Start value sweep
	1. Branch: experiment/exp203-lr-start
	2. Configurations: 7 start values in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
	3. Runs: 25 seeds per configuration (175 runs total)
    

### 2.3.2. Lookahead Optimizer Sweep

We next varied Lookahead hyperparameters to see whether altering the synchronization frequency and interpolation strength could accelerate convergence.
- Branch: experiment/exp301-lookahead
- Configurations:
	- k (slow update frequency): [3, 5, 7, 10]
	- α (interpolation strength): [0.590 (low), 0.904 (high)] with a cubic schedule
	- Constant-schedule baseline: k = 5, α = 0.5 (no decay)
	- Ablation: Lookahead disabled (SGD only)
- Runs: 25 seeds per configuration (≈250 runs total)
### 2.3.3. Test-Time Augmentation (TTA) Ratio Sweep

Finally, we tuned TTA ratios at evaluation time, holding the training procedure fixed at the baseline configuration and using TTA level 2.

- Branch: experiment/exp401-tta-weights
- Configurations: 9 weight ratios of (untranslated : translated) images, from 0.30:0.70 to 0.70:0.30 in steps of 0.05
- Implementation: Modified `infer_mirror_translate()` to accept weights as parameters.
- Runs: 25 seeds per configuration (225 runs total)  

Because TTA is applied only at test time, these sweeps are intended to improve accuracy at roughly fixed training cost.

## 2.4. Epoch-Count Sweep 

After identifying the best-performing configurations from the LR, Lookahead, and TTA sweeps (i.e., those that match or exceed baseline accuracy), we conducted a follow-up sweep over the number of training epochs.

We varied the effective epoch count within a narrow range around the baseline (e.g., 9.1–9.9 epochs, with ~25 seeds per setting) to determine how far we could shorten training while maintaining a mean accuracy of at least 94%. For settings that improved accuracy (such as favorable TTA ratios), we used the extra margin above 94% to justify more aggressive reductions in epochs.

## 2.5. Combined Speedrun Experiments

In the final stage, we combined the best settings from the previous sections (LR schedule, Lookahead parameters, and TTA ratio) into a single “combined” configuration. We ran this configuration using the same protocol as the baseline (same number of runs, same hardware, same logging) to test whether the individual improvements compose additively, as suggested in the CIFAR-10 speedrun paper. We then compared the combined method’s time to achieve 94% accuracy and its accuracy distribution against the baseline (rerun for 1,000 runs for better results) using our logged statistics, based on 1,000 runs.

## Total experimental budget:

- Total runs for every experiment: 3,350 runs
- Total GPU time: 5.7 A100-hours
- Per-run cost: ~0.0008 A100-hours (~4 seconds)

We run many random seeds per configuration for three reasons:

1. Stability of estimates: Multiple seeds let us estimate a distribution of accuracy and runtime, not just a single noisy number.
    
2. Confidence intervals and significance: With sufficient runs (e.g., 25 for screening, 1,000 for final comparisons), we can compute tight 95% confidence intervals and determine whether performance differences are statistically significant or just noise.
    
3. Fair comparison between methods. By assigning the baseline and final combined configuration the same number of runs, we ensure that any improvements (or regressions) in accuracy or time are due to the method itself, rather than the noise.
    

# 3. Results

## Summary of Best Configurations by Method

|Experiment (branch)|Description|Runs|Mean acc. (95% CI)|Mean time (95% CI)|Success rate (≥94%)|Notes|
|---|---|---|---|---|---|---|
|Exp000-baseline<br><br>(lr start 0.20, peak 0.23, end 0.07, TTA 0.5/0.5)|Original training setup (CIFAR-10, A100)|1000|94.01% ± 0.01%|4.166 s ± 0.002 s<br><br>(excluding first run)|52.2% <br><br>(522/1000)|Baseline reference|
|exp230-lr-end-sweep (end = 0.05)|Lower LR end value to 0.05|25|94.04% ± 0.06%|4.04 s ± 0.05 s|68% (17/25)|Accuracy slightly increases. Time improves significantly from baseline.|
|exp210-lr-peak-sweep (peak = 0.15)|Earlier LR peak position|25|94.04% ± 0.06%|4.11 s ± 0.02 s|56% (14/25)|Marginal time increase; However, accuracy/time CIs overlap; not implemented|
|exp220-lr-start-sweep (start = 0.30)|Higher LR start value|25|94.01% ± 0.05%|4.10 s ± 0.02 s|60% (15/25)|Marginal accuracy increase; accuracy/time CIs overlap; not implemented|
|exp310-lookahead-sweep (baseline cfg)|Lookahead with k = 5, decaying α schedule|25|93.99% ± 0.05%|4.09 s ± 0.03 s|44% (11/25)|Alternative Lookahead variants & SGD-only are all worse|
|exp410-tta-weights-sweep (0.30 / 0.70)|TTA level 2 with 30% untranslated / 70% translated|25|94.09% ± 0.05%|Untracked|76% (19/25)|Accuracy increase is statistically significant|
|Exp420-early-stop-baseline (epoch 9.8)|Implement early stopping on baseline|25|94.02%|3.88 s|56% (14/25)|Time decrease is significant.|
|Exp421-early-stop-optimal-lr (epoch 9.7)|Implement early stopping on optimal LR|25|94.02%|3.91 s|56% (14/25)|We decided to use epoch 9.7 as the final epoch.|
|Final result (9.7 epoch lr end 0.05 TTA 0.3/0.7)|Final experiment on the main branch|1000|93.98% ± 0.01%|4.082 s ± 0.002 s|42.5%<br><br>(425/1000)|Statistically significant. However, the accuracy didn’t meet 94%.|

  

# 4. Discussion and Future Work 

## 4.1. Discussion

#### Baseline configurations

The baseline configuration (Exp000) achieves 94.01% ± 0.01% accuracy with a mean runtime of 4.166 s ± 0.002 s over 1000 runs. This gives a very tight estimate of the “true” performance and serves as a solid reference for all later changes.

#### Effect of learning rate schedule

The three LR sweeps (Exp210/220/230) all have mean accuracies around 94.0–94.04% with 95% CIs that overlap the baseline. This means none of the LR variations can be claimed to clearly improve accuracy.

However, Exp230 (lower end LR = 0.05) yields 4.04 s ± 0.05 s, whose CI does not overlap the baseline time CI. We can then be confident that it reduces runtime without compromising accuracy in a statistically significant manner. The other LR variants show only marginal time changes and are therefore not promoted.

Overall, the LR experiments suggest that we can safely adopt a slightly lower end LR to gain speed, but any accuracy differences at this stage are within noise.

#### Lookahead vs. SGD

The Lookahead sweep (Exp310) slightly reduces the success rate (44% vs. 52.2% baseline) and achieves 93.99% ± 0.05% accuracy with a similar runtime. Given that both accuracy and success rate are worse and the intervals overlap strongly with baseline, Lookahead introduces extra complexity without measurable benefit, so it is rejected.

#### Test-time augmentation (TTA) and untranslated weight

The TTA sweep with 0.30 untranslated / 0.70 translated (Exp410) reaches 94.09% ± 0.05% accuracy and the highest success rate among the 25-run experiments (76%). Compared to the implicit baseline of 0.5/0.5, this suggests that giving more weight to translated views helps.

Intuitively, the model has been trained heavily on augmented/translated images, so at test time it may benefit from seeing more of the distribution it was optimized on. Keeping some untranslated weight (30%) stabilizes predictions on clean inputs, while emphasizing the augmented views (70%) may reduce variance and better exploit the model’s invariances. A very low untranslated weight might be detrimental if we proceeded further (e.g., 0.1/0.9), but 0.3/0.7 appears to strike a good balance in the small-scale experiment.

#### Early stopping and cumulative effects

Early stopping at epochs 9.8 and 9.7 (Exp420/421) shows that we can shorten the training time to ~3.9 seconds while maintaining an accuracy of around 94.02% in 25 runs. Again, the 95% CIs are wide and overlap baseline, so these results are consistent with “no real change” in accuracy and a meaningful speedup.

#### Final configuration results

The final configuration combines several tweaks proven successful in the parameter sweep experiments: lowering the end LR to 0.05, using TTA 0.3/0.7, and early stopping at epoch 9.7. With 1,000 runs, this setup achieves 93.98% ± 0.01% accuracy, a runtime of 4.082 s ± 0.002 s, and a 42.5% success rate. The runtime improvement is statistically significant compared with the baseline (4.166 s ± 0.002 s), resulting in an approximately 2% reduction in runtime.

However, the accuracy now falls slightly just below 94%, and its 95% CI no longer overlaps with the baseline CI. In the smaller 25-run experiments, the standard error was much larger, so the confidence intervals were wide enough that 94% could still be inside them even if the true mean was slightly lower. Scaling to 1000 runs drastically shrinks the standard error, tightening the CI (±0.01%) and revealing that the combined configuration is in fact slightly worse than the baseline in accuracy. The 25-run experiments were therefore useful for screening, but not precise enough to guarantee that the final stacked tweaks would still meet the 94% target.

## 4.2. Future Experiments

To resolve the remaining questions and possibly recover the baseline success rate while keeping the speedup, we can:

1. Test 9.8 epochs with the final LR + TTA config  
    Run a 1000-seed experiment at epoch 9.8 with LR end = 0.05 and TTA 0.3/0.7, and check whether this slightly longer training recovers the 94% accuracy threshold while keeping the mean time close to the improved 4.082 s.  

2. Refine TTA weights around 0.3/0.7  
    Try nearby weightings, such as 0.4/0.6 and 0.2/0.8, with sufficient runs (e.g., 100–200) to see if there is a “sweet spot” that boosts accuracy without a significant time cost.  

3. Revalidate promising configs with larger n before finalizing  
    For any configuration that looks better on 25 runs, repeat it with at least n = 200 before promoting it to the 1000-run main experiment to reduce surprises when scaling.
