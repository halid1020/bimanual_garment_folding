# Cover Letter

**To:** The Editor-in-Chief, IEEE Transactions on Robot Learning

**Manuscript:** *LaGarNet: Latent World Models for Quasi-Static Pick-and-Place Garment Flattening*

**Type:** Regular Paper

---

Dear Editor-in-Chief,

We are pleased to submit our manuscript, *LaGarNet: Latent World Models for Quasi-Static Pick-and-Place Garment Flattening*, for consideration as a Regular Paper in the *IEEE Transactions on Robot Learning*. This work presents a goal-conditioned recurrent state-space model that learns the latent dynamics of pick-and-place garment manipulation, demonstrating that a single policy built upon this framework can successfully flatten four distinct garment types in both simulation and the real world.

Latent world models are predominantly studied under velocity or position control, where consecutive observations are temporally continuous. In contrast, high-level pick-and-place manipulation inherently involves large, discontinuous deformations, rendering successive observations effectively disjoint. Whether a recurrent state-space model can effectively capture dynamics in this regime has remained an open question, particularly as prior mesh-free planners have struggled to match the performance of mesh-based alternatives for garment flattening. Our central finding is that such models can indeed succeed, provided the latent inference process is explicitly goal-aware.

Our manuscript presents three key contributions. First, we introduce a **goal-conditioned recurrent state-space model (GC-RSSM)** that conditions both the prior and posterior latent states on the goal observation, rather than limiting this conditioning to a downstream actor or critic. Second, we propose a **data-collection strategy for offline pick-and-place flattening** that combines a Diffusion Policy expert—learned from a minimal set of human demonstrations—with a mask-biased random policy. This approach effectively replaces the hand-engineered oracles relied upon in previous literature. Third, we formulate a **coverage-alignment reward** computed entirely from top-down segmentation masks, ensuring the exact same reward function can be utilised in both simulation and on real hardware without requiring particle-level state information.

We believe these results will be of significant interest to the readership in terms of both capability and computational efficiency:

* **Capability:** Our single policy exceeds 95% normalised coverage across all four simulated garment types within 20 action steps. Furthermore, it transfers zero-shot to a physical UR5e robotic arm, achieving 90.3% normalised coverage, 80.7% normalised improvement, 81.1% maximum IoU, and 27 successes out of 40 physical trials within 20 actions (compared to 9/40 for the PlaNet-ClothPick baseline under identical conditions).
* **Cost & Efficiency:** LaGarNet matches the performance of the state-of-the-art mesh-based planner, MEDOR, while requiring significantly fewer parameters (7.11 M compared to 37.5 M) and drastically reducing planning time (approximately 6 s per action compared to 103 s). Six seconds per action is affordable on a physical robot where a hundred is not; we are careful to add that the full perception–action cycle still takes about 24.5 s per step, and Section IV-D names this as a critical limitation for deployment rather than a solved problem.

We also provide a candid discussion of the method's current limitations. Notably, because the system lacks a learned stopping criterion, it reaches a success state far more frequently than it holds one: in the simulated longsleeve arena, 76.7% of trials reach success within 20 steps but only 13.3% are still successful at the final step. We evaluate this transparently within the manuscript rather than relying solely on cumulative metrics.

Three points are worth stating plainly rather than leaving the reviewers to rediscover them. First, we have not added a parameter-matched model-size study: Table I is a deployment-cost comparison of the published configurations, and our efficiency claim rests on inference time, which follows from planning in a compact latent space rather than over a reconstructed mesh. Second, hybrid action primitives are outside this paper's scope; we have withdrawn the earlier unsupported claim about their cost rather than defend it with a study the paper does not need. Third, the reliance on a goal image is weaker than it may appear — the goal is canonical, captured once per garment and reused, and independently augmented during training — but we have not run a canonical-versus-generated comparison, and the manuscript records that as out of scope.

An earlier iteration of this work was reviewed by the *IEEE Transactions on Robotics* (submission 26-0591). We are highly appreciative of the reviewers' constructive feedback, which has informed substantial revisions to this current submission. The most significant changes include:

1. A new ablation study over the MPC planning horizon, reporting both performance and per-action runtime (Figure 5(f), Section IV-C).
2. The inclusion of paired *Max* and *Last* metrics throughout Figure 3(a), allowing cumulative and final-step behaviour in simulation to be evaluated independently. The real-world table (Table II) deliberately remains cumulative: each cell averages only ten physical trials, so a single final-step mean would be noisy, and Figure 6(b) measures end-state behaviour more directly.
3. A restated MEDOR comparison evaluated under matched observation cropping, explicitly described as such.
4. The removal of the cognitive-psychology subsection in favour of a detailed cross-entropy-method planning recipe.
5. A structural reorganisation that moves the architecture, hyperparameters, baseline formulations, and the real-robot setup into the appendices to ensure the main body adheres to the page limit.

Alongside the manuscript, we have attached supplementary video material demonstrating real-world rollouts of LaGarNet flattening each of the four garment types. This is accompanied by four appendices detailing: (A) network architecture, planning, and hyperparameters; (B) the Diffusion Policy formulation and data augmentation; (C) the simulated garment distribution; and (D) the real-robot setup, calibration, segmentation, and grasping procedures.

This manuscript represents original work, has not been published elsewhere, and is not currently under consideration by any other journal or conference.

Thank you for your time and consideration.

Yours sincerely,

Authors