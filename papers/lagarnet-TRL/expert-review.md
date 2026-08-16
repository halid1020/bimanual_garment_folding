# LaGarNet — an expert reviewer's assessment

*Written as if I were a senior robot-learning reviewer handling this submission for T-RL, with full
access to the manuscript, the figures, the run logs and the response documents. It is deliberately
unflattering where being flattering would not help. Numbers are taken from `main.tex` and, for the
horizon ablation, re-derived from `/home/halid/lagarnet_data` via notebook cell 4.*

**Overall recommendation: minor-to-major revision, leaning accept.** The core empirical claim is
sound, well ablated and honestly reported. What holds it back is not the result but its framing:
the paper's own evidence points somewhere slightly different from where the title and abstract
point, and the strongest baselines never meet the robot.

---

## 1. What the paper actually establishes

A goal-conditioned RSSM, planned greedily with CEM over a mask-computable reward, flattens four
garment topologies with one policy, in simulation and zero-shot on a UR5e. It matches MEDOR at
5× fewer parameters and 17× faster inference, and reaches 90.3 % NC / 80.7 % NI / 81.1 % Max IoU
with 27/40 successes in 40 physical trials.

That is a real result, and the negative-space claim matters as much as the positive one: **no
mesh-free planner had previously matched a mesh-based one on garment flattening.** This one does,
on the same garments, physics and initial states. Reviewers should not lose that in the noise about
horizons.

## 2. Strengths

**The honesty is unusual and should be rewarded.** §IV-C ends by ranking the sources of performance
and puts the authors' own headline contribution *last*: "Mask-constrained action sampling matters
most… The reward comes next… The architecture contributes least on its own." Most submissions bury
that. Publishing it is the single most useful thing in the paper for anyone who wants to build on it.

**The ablations are the real contribution, not the architecture.** Six axes — latent dynamics,
reward, data size, data composition, planning horizon, action constraint — each with a clean
counterfactual. The unconstrained-CEM control at H = 1 (0.0 % SR, 51.0 final IoU against 73.3 % and
71.5) is the most informative single number in the paper, and it is an argument *against* the
authors' own emphasis. Good.

**The reward is the quietly excellent bit.** Coverage + Max-IoU deltas from top-down masks alone,
identical in simulation and reality, no particle state, no learned smoothness net. It is not clever,
and that is the point: it is the reason a simulation-trained policy transfers without a tuned digital
twin. The reward ablation (§IV-C: only SFA comes close) earns the claim.

**Sim-to-real is done properly.** Same reward, same observation pipeline, no fine-tuning, a workspace
heuristic that is described rather than hand-waved, and a full failure taxonomy with rates. The
step-wise/trajectory failure table (Figure 6(b)) is more useful than most papers' entire evaluation.

**The limitations section is genuinely a limitations section.** It names the sim-to-real gap, the
multi-task capacity problem (underfits longsleeves and trousers, overfits skirts and dresses), the
front/back blindness, the goal-image assumption, and the 24.5 s per step. It does not hedge.

## 3. Weaknesses

### 3.1 The best horizon is 1, in a paper about world models

This is the objection that will be raised again, and the current answer is good but incomplete.
Success falls monotonically 73.3 → 63.3 → 53.3 → 36.7 → 30.0 % as H goes 1 → 5. So the learned
dynamics are being used as a **one-step action scorer**, not for planning in any meaningful sense.

The paper's defence — one primitive causes one large deformation, so error compounds — is plausible,
and Appendix A.4 adds a sharper reason that deserves promotion into the main text: *mask rejection
only constrains the first action of a plan*, because the cloth state after the first pick-and-place
is unknown. That is a **structural** explanation, not a property of the world model, and it means the
horizon result partly measures the constraint mechanism rather than the dynamics. Say so plainly.

What would settle it: score H > 1 rollouts with the constraint applied to every step using the
model's own predicted mask. If performance still falls, the compounding-error story is confirmed. If
it does not, the current conclusion is an artefact of the sampler. This is a small experiment with a
large payoff and I would ask for it.

### 3.2 The strongest baselines are never run on the robot

Real-world comparison is against PlaNet-ClothPick and a human only. MEDOR is excluded for runtime,
Diffusion Policy for workspace-constraint incompatibility. Both reasons are honest and both are
stated — but the paper's headline is real-world deployment, and 9/40 for PlaNet-ClothPick is a weak
bar. A reader cannot tell from this paper whether LaGarNet beats a *good* alternative in reality or
only a related one.

The Diffusion Policy exclusion is the more troubling of the two, because §V argues the workspace
constraint is precisely what makes LaGarNet transferable — which makes the untested comparison the
paper's own central claim. Even one garment type, ten trials, with the DP given its best available
treatment, would move this from asserted to demonstrated.

### 3.3 The gap to a human is large and under-discussed

27/40 versus 38/40, under an *identical* single-arm constraint, with relaxed real-world thresholds
(NC ≥ 85, Max IoU ≥ 75, down from 90/80). Per garment the picture is starker: 10/10 on dresses and
skirts, 4/10 on longsleeves, 3/10 on trousers. The unified policy is close to solved on simple
topologies and roughly a third as good as a person on complex ones.

The paper attributes this to multi-task capacity, which is reasonable, but the natural control —
task-specific real-world policies on longsleeves and trousers — is absent, and the simulation result
points the other way (all-garment beats task-specific there). That inversion between simulation and
reality is interesting and is not discussed. It may be the most publishable loose thread here.

### 3.4 The 0/10 out-of-distribution result

Zero successes on the soft longsleeved dress, with 77.2 NC. The response document worries this
"will lead someone to simply say it does not work". As a reviewer: it does not read that way to me,
*because* it is reported alongside the coverage. What would read badly is the version where the OOD
garment is folded into the headline. Keep it separate and keep the zero.

But do draw the right conclusion from it. The framework's stated assumption is that geometry
dominates and material properties are secondary. This result is evidence against that assumption,
not merely a hard case. §IV-D says so in passing; it deserves a sentence in §V.

### 3.5 Deployment cost is framed as solved when the paper knows it is not

The abstract and §I lead with 17× faster inference. §IV-D concedes that a real step takes ~24.5 s,
of which planning is 7.05 s — perception (8.19 s) and execution (9.20 s) dominate. A 20-step episode
is over eight minutes. The efficiency claim against MEDOR is legitimate and well evidenced; the
implicit claim that this makes the system deployable is not, and the paper is better than that. The
two should sit in the same paragraph, not eighteen pages apart.

### 3.6 Metrics cannot see what the task is for

NC and Max IoU are top-down projections. They cannot distinguish a flat garment from one with a
hidden fold underneath, cannot see layer ordering, and by the authors' own account cannot see
front/back. The failure table shows "Fold Underneath" 3/40 and "Cannot untwist" 5/40 — i.e. the
metric is demonstrably blind to failures the system actually commits. The paper acknowledges this.
It would cost little to add one grounded number, e.g. particle-level flatness in simulation, to
calibrate how far the projected metrics can be trusted.

### 3.7 Novelty, honestly assessed

GC-RSSM = RSSM + goal in the prior and posterior. CEM planning is standard. The reward is a linear
blend. Diffusion-policy-assisted data collection is a sensible recipe, not a new principle. The
paper says as much, which defuses the objection but does not eliminate it: this is a strong systems
and empirical contribution, not an algorithmic one. For T-RL that should be acceptable — but the
paper should own it in the abstract rather than only in §V, because the abstract currently reads as
though GC-RSSM is the finding, when the finding is *that the whole recipe works in a regime where it
was not known to*.

## 4. Questions I would put to the authors

1. If mask rejection were applied at every step of an H > 1 rollout using predicted masks, does the
   horizon result survive? This determines whether §IV-C measures the dynamics or the sampler.
2. All-garment beats task-specific in simulation, yet the real-world failures are attributed to
   multi-task capacity. What reconciles these?
3. Why does the real-world protocol relax the success criterion (85/75 vs 90/80)? What are the
   real-world numbers under the simulation thresholds? A reader will assume the worst otherwise.
4. Figure 5(f) uses `iou_thresh = 0.79` where §IV states 0.80. Presumably a float-tolerance guard —
   but state it, or align it.
5. The reward predictor is described as least reliable once the garment is flat, which is the direct
   cause of "Messing-near-success" (168/298). Was a simple fix tried — e.g. refusing any action with
   predicted reward below zero? That is a no-op stopping rule available today.
6. How sensitive is transfer to the canonical goal image? One perturbation study (goal captured on a
   different day, different pose) would convert an assumption into a measurement.

## 5. What would move this to a clear accept

- The per-step-constrained H > 1 experiment (§3.1). Small, decisive.
- Any real-world number for a second strong baseline, even a partial one (§3.2).
- Real-world results under the simulation success thresholds, reported alongside the relaxed ones.
- One paragraph reconciling the simulation/reality inversion on all-garment versus task-specific.
- Abstract and introduction adjusted so the deployment-cost claim carries its own caveat.

None of these requires a new method. All are within reach of the existing infrastructure.

## 6. What I would not ask for

- A parameter-matched model-size study. The authors decline it and they are right: MEDOR's cost is
  test-time mesh fitting, not network size, so matching parameters would not change the comparison.
- Hybrid primitives. Out of scope, correctly, and the unsupported complexity claim has been withdrawn.
- Generative goal images. A different paper.
- AUC metrics. Max/Last pairs already carry the information for the question being asked.

## 7. The one-line verdict

*A carefully evaluated, unusually honest systems paper that demonstrates something previously
unshown — a mesh-free latent planner matching mesh-based planning on garment flattening, and
transferring zero-shot — but whose framing oversells the world model relative to its own ablations,
and whose real-world comparison is too thin to support the deployment story it tells.*
