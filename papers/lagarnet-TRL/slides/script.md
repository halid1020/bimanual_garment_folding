# LaGarNet — presentation script

Narration for `lagarnet-5min.html`. One section per slide, in slide order. `build_video.py` parses
this file: it reads the `## NN —` headings and takes the plain paragraphs beneath each as the text
to synthesise, so **keep the heading format and do not add prose outside the blockquote-free
paragraphs**. Bullet lists, tables and HTML comments in this file are ignored by the parser.

Register is neutral spoken academic English. Every figure quoted here appears in `main.tex`.
No author names and no institutions are spoken — the deck is recorded with `?hide-authors`.
The closing references slide has no section here on purpose: any deck slide past the last
script section is held silently at the end of the video.

Voice: the author's own recording, `my_narration/slide-N.ogg`, one file per section below.
Running time ~8:27. `build_video.py --voice tts` re-synthesises the script with edge-tts
(`en-GB-RyanNeural`) instead, which is how the earlier 7:26 cut was made.

---

## 01 — Title

Hello everyone. Today, I will be presenting LaGarNet: a latent world model designed for quasi-static, pick-and-place garment flattening. We show that a single policy trained entirely in simulation flattens four distinct garment types on real hardware, with no further fine-tuning.

## 02 — The task

Our primary task is garment flattening, a critical prerequisite for downstream tasks like folding, ironing, and pressing. This task is inherently challenging because cloth possesses vast degrees of freedom, undergoes severe non-linear deformations, and exhibits high self-occlusion from a top-down perspective. To rigorously evaluate the underlying representation rather than relying on hardware complexity, we deliberately constrain our physical setup to a single robotic arm, a single pick-and-place primitive, and one top-down RGB camera.

## 03 — Related work and research question

Existing data-driven flatteners generally fall into three families. Behaviour cloning is conceptually straightforward, but adequately covering the vast state space of crumpled configurations requires an impractical volume of human demonstrations, often necessitating scripted oracles. Model-free reinforcement learning explores this space directly, yielding general flatteners, but suffers from poor sample efficiency and cannot predict the forward effects of actions. Model-based methods remain rare in garment manipulation; standard latent models tend to produce blurred reconstructions that lose the critical edge and corner features necessary for shaping. This leads to our core research question: can latent world models built for continuous control still function in quasi-static primitive setups, where consecutive observations are not temporally continuous?

## 04 — Goal-conditioned recurrent state-space model

Our solution is LaGarNet, built upon a goal-conditioned recurrent state-space model. The fundamental design choice here is the direct integration of goal-conditioning into the latent inference process itself. Both the prior and posterior distributions over the latent state are explicitly goal-aware. A recurrent backbone maintains a belief state over the occluded regions of the garment, while a joint decoder predicts both the top-down mask and the task reward. During inference, we plan using the cross-entropy method, sampling candidate pick-and-place actions and ranking them based on the predicted reward. Ablation studies reveal that removing goal-conditioning from the latent inference costs twenty-two and a half percentage points of success rate, from seventy-five percent down to fifty-two and a half.

## 05 — Data collection

Our second contribution is the data collection paradigm. We train a Diffusion Policy expert on just fifty human demonstrations, allowing it to reliably reach near-flattened states that a random exploration policy would rarely discover. We then mix this expert's rollouts in a one-to-one ratio with a mask-biased random policy, which drives state exploration by uniformly sampling pick points exclusively from the garment's segmentation mask. This approach yields a robust dataset of five thousand episodes without relying on complex scripted oracles, creating a paradigm that is agnostic to cloth dynamics and likely to generalise to other deformable object tasks.

## 06 — Coverage-alignment reward

Our third contribution is a novel coverage-alignment reward. This metric jointly evaluates state coverage and alignment, computed entirely from top-down binary segmentation masks. Crucially, it eliminates the need for privileged particle-level state or learned smoothness estimators. This allows the exact same continuous and linear reward function to be evaluated in simulation and directly on physical hardware, making it highly amenable to predictive learning.

## 07 — Simulation results

In simulation, LaGarNet achieves parity with MEDOR, the state-of-the-art mesh-based planner, whilst outperforming both PlaNet-ClothPick and Diffusion Policy across short and medium planning horizons. Furthermore, a single, unified policy trained across all garments outperforms specialised, single-garment variants, suggesting that morphological diversity acts as a beneficial regulariser. Ultimately, the policy exceeds ninety-five percent normalised coverage across all four simulated garment categories within twenty action steps.

## 08 — Computational cost

This comparison also has a crucial computational dimension. LaGarNet requires five times fewer parameters than MEDOR and plans seventeen times faster, with no loss in flattening quality. Six seconds of planning per action is affordable on a physical robot, where one hundred and three seconds is not. We are careful not to overstate this: with perception and robot execution included, a single real manipulation step still takes about twenty-four seconds, so the full perception-action cycle remains the deployment bottleneck.

## 09 — Real-world rollouts

We subsequently transferred this simulation-trained policy directly to a physical UR5e robotic arm, achieving this with zero real-world fine-tuning. The videos shown here demonstrate rollouts on all four garment types, accelerated to ten times real speed. The policy operates solely on a top-down observation and the goal image, and the reward is computed from the exact same segmentation masks utilised in simulation. This consistency is precisely what enables successful zero-shot transfer without requiring a highly tuned digital twin of the physical cell.

## 10 — Real-world results

Quantitatively, across forty physical trials of twenty actions each, LaGarNet achieves ninety point three percent normalised coverage, eighty point seven percent normalised improvement, and an eighty-one point one percent maximum intersection over union. It successfully completes twenty-seven out of forty trials, against nine out of forty for the PlaNet-ClothPick baseline, though a human operator still reaches thirty-eight out of forty. Success here means reaching eighty-five percent coverage and seventy-five percent maximum intersection over union at any point within a twenty-step episode; we relax these thresholds from their simulation values to allow for the sim-to-real gap. We also stress-tested a fifth garment, a soft long-sleeved dress whose topology is absent from the simulated training set. Evaluated separately over ten further trials, the policy recovers most of its area, at seventy-seven percent coverage, but never meets the success criterion, scoring zero out of ten.

## 11 — Lessons and future work

We draw two primary lessons from this work. First, latent world models can be highly effective in quasi-static manipulation regimes, provided that goal-conditioning is integrated directly into the latent inference process. Second, greedy, one-step planning yields the highest performance. Because a single manipulation primitive induces massive state deformation, predictive error compounds faster than extended lookahead can provide value—success drops from seventy-three point three percent at horizon one, to thirty percent at horizon five. The principal limitation of our current system is the lack of an explicit stopping criterion. Immediate future work includes integrating a learned termination condition and extending the framework to goal-directed folding. Looking ahead, we view this formulation as a promising pathway toward general deformable-object manipulation. Thank you for listening.