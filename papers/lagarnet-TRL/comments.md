# LaGarNet — reviewer comments and how the T-RL manuscript resolves them

This is a working document for the T-RL resubmission. The 28 comments below are the reviewers' original
text, numbered in the order they appeared. Every revision made in response is highlighted **in
blue** in `main.tex` via the `\rev{}` macro; setting `\showrevfalse` in the preamble produces an
identical but unhighlighted camera-ready copy.

**Status counts: 21 resolved, 6 partially resolved, 0 not resolved, 1 not planned.**

---

## 1. Numbering changed between submissions

The reviewers refer to T-RO numbering. The same objects now carry different numbers:

| Reviewer's reference (T-RO) | Same object in T-RL |
|---|---|
| Table I — hyperparameters | **Table III**, Appendix A |
| Table II — simulation configuration | **deleted**; parameters inlined in §IV-A1 |
| Table III — longsleeve flattening | **Figure 3(a)**, now a heatmap with Max and Last side by side |
| Table IV — parameters and runtime | **Table I** |
| Table VI — real-world flattening | **Table II** |
| Figure 9 — real-robot parameters | **Figure 8(c)** (`tab:seg-para`) |

## 2. Structural changes relative to T-RO

T-RO had no appendices; the T-RL body ends at the bibliography and everything after it is an
appendix, so those pages fall outside the 12-page limit.

| Change | Detail |
|---|---|
| Deleted | §III-A "Inspiration from Cognitive Psychology" (comment 2) |
| Deleted | Table II, the simulation-configuration table (comment 5); its parameters now sit in the §IV-A1 prose |
| Moved to Appendix A | Network architecture and hyperparameters |
| Moved to Appendix B | Diffusion Policy formulation and data augmentation |
| Moved to Appendix C | Simulation garment distribution |
| Moved to Appendix D | Real robot setup, calibration, segmentation, grasping, workspace heuristic |
| New | Appendix A.4, "Test-Time Planning with the Cross-Entropy Method" (comment 10) |
| New | Figure 5(f), MPC planning-horizon ablation (comments 4, 9, 18, 28) |
| Rewritten | Title and abstract (comments 16, 24) |
| Renamed | §III-D "Data Collection with Proximal Expert and Random Policies" → "Proximal Expert and Random Policy Data Collection" |

Section headings are deliberately **not** highlighted in blue, because colour inside `\section{}`
corrupts the hyperref PDF bookmarks. The renames are recorded here instead.

---

## 3. Resolution table

| # | Gist | Reviewer comment | Status | Where in T-RL |
|---|---|---|---|---|
| 1 | Distinguish algorithmic novelty from system integration | Many components are adaptations or combinations of existing ideas: RSSM/PlaNet-style latent dynamics, goal conditioning, CEM action selection, reward shaping based on coverage/alignment, Diffusion Policy-assisted data collection, and workspace masking. These pieces may be useful together, but the paper should more carefully distinguish between algorithmic novelty and system integration. | Resolved | §I ¶4; §V ¶1; contribution 1 |
| 2 | Remove the cognitive-psychology subsection | Section III-A, which motivates the method through cognitive psychology, is weakly connected to the actual algorithm. I recommend removing this section or reducing it to a short motivating paragraph. The space would be better used to clarify the actual test-time planning procedure, data-generation protocol, and evaluation metrics. | Resolved | Deleted; space reused for §III CEM recipe and §IV metrics |
| 3 | Typographical and metadata errors | The manuscript contains many typographical and grammatical errors that should be corrected before publication. Examples include 'In this paper we shot,' 'probablisitic,' 'continous,' 'oringal,' 'readuces,' 'server self-occlusion,' and 'all model.' There are also reference and metadata errors, such as 'WTO' where the cited organization appears to be WHO. | Resolved | All seven named typos and the WHO metadata fixed; the five errors the revision introduced now corrected (§I ¶4, §IV-A2, §IV-C, Table II caption, §V ¶1) |
| 4 | Horizon ablation H=1,2,3,5 with performance and runtime | The paper should either add an ablation over MPC horizons, e.g., (H=1,2,3,5), reporting both performance and runtime, or explicitly justify why (H=1) is chosen. Possible explanations might include compounding model error, the dense step-wise reward, quasi-static PnP dynamics, or computational constraints. Without such evidence or explanation, claims about multi-step planning or long-horizon reasoning should be weakened. | Resolved | Figure 5(f); §IV-C |
| 5 | Evaluate MEDOR under matched conditions, or call it contextual | The authors should either evaluate MEDOR and LaGarNet under the same garment geometry, physics, camera, initial states, and success criteria, or present the MEDOR comparison as contextual rather than a direct state-of-the-art match. | Resolved | §IV-A2; Figure 3(a) caption |
| 6 | Final-step, first-hit, success-maintenance and AUC metrics | The authors should additionally report final-step metrics, first-hit success, and a success-maintenance metric, for example whether the garment remains successful for one or more subsequent actions. Area-under-curve metrics would also help characterize sample efficiency and stability across the full rollout. | **Partial** | Figure 3(a) pairs Max with Last; §V ¶5. No AUC |
| 7 | Table III caption: is SR cumulative? why 20 vs 30 steps? | Table III reports blocks for 5, 10, 20, and 30 steps, but the caption does not clearly state whether SR means success exactly at that step or success achieved at least once within the first (N) actions. Based on the metric definition, the intended interpretation seems to be cumulative performance up to the horizon, but this should be explicitly stated. The paper should also clarify why dataset trajectories are described as up to 20 steps while Table III reports 30-step evaluation. | Resolved | §IV opening; captions of Figure 3(a) and Table II |
| 8 | Parameter-matched or model-size ablation | Table IV reports that LaGarNet has fewer parameters than the Diffusion Policy baseline and MEDOR. This is useful, but it is a static comparison, not a controlled model-size study. If the authors want to claim architectural superiority or parameter efficiency more broadly, they should include a parameter-matched comparison or a small/medium/large model-size ablation for LaGarNet and the main neural baselines. | **Not planned** | Declined: Table I is presented as a deployment-cost comparison, not a claim of architectural parameter efficiency (§5) |
| 9 | Ablations omit the planning horizon | The ablations examine latent dynamics, reward design, data size, and data composition, but they do not test the planning horizon. Given that the central distinction of the paper is the use of a learned world model for action selection, this is a major omission. A horizon ablation would also help determine whether the learned latent dynamics are actually useful for multi-step prediction/control or mainly useful as a one-step action scorer. | Resolved | §IV-C; Figure 5(f) |
| 10 | Specify the exact test-time procedure | The paper should more clearly specify the exact test-time procedure: how CEM initializes and updates the action distribution, how many elite samples are used, how workspace and cloth-mask constraints are imposed during sampling, whether action candidates are sampled in normalized image coordinates or physical coordinates, and how the selected pixel-space action is converted into robot motion. The current description gives the high-level idea but is not sufficient for reproduction. | Resolved | §III-B; Appendix A.4 |
| 11 | Figure 9 shows (r_n, rf) = 0.7 m, 0.2 m — inverted, and notation | Figure 9 lists (r_n, rf = 0.7m, 0.2m), which appears inverted because the far radius should normally be larger than the near radius, and the notation should be r_f. The authors should check this parameter table carefully. | Resolved | Figure 8(c) |
| 12 | Table III is dense; put final-step metrics next to the maxima | Table III is dense and hard to parse. The authors should explicitly state that the rows correspond to evaluation horizons and clarify whether the metrics are cumulative maxima or exact-step values. If max-over-trajectory metrics are retained, final-step metrics should be placed next to them. | Resolved | Figure 3(a); Table II caption |
| 13 | Verify every citation number, author list, venue and year | There are several apparent citation-numbering or metadata inconsistencies. For example, the text refers to VCD and MEDOR in ways that do not always match the listed reference numbers. Before publication, the authors should systematically verify that every citation number, author list, title, venue, and year is correct. | Resolved | Done in reference |
| 14 | Provide a video with uncut rollouts | If the authors intend to provide a video, it would be useful to include uncut rollouts showing both successes and failures, especially near-success disturbance cases, one-step action selection behavior, and real-robot timing. A video would be particularly important because the paper's main claims concern real-world garment flattening quality and deployment. | **Partial** | Videos submitted and the narrated slide video built; §I points to them. The first-page `\thanks` footnote is **not** in the manuscript, and the footage is a 10x highlight cut rather than the uncut real-speed rollouts asked for — see comment 14 below |
| 15 | Component novelty is limited; the work is system integration | The novelty of the individual components is limited. RSSM-style latent dynamics, goal conditioning, CEM planning, diffusion policy data collection, and coverage-based rewards are all natural extensions of existing methods. The paper's main effort is system integration rather than a clearly new algorithmic principle. | Resolved | Same as #1 |
| 16 | Weaken the "human-level" claim in title and abstract | The claim of 'human-level' garment flattening is also not well supported. In real-world experiments, human operators still substantially outperform LaGarNet in success rate and final metrics. The method also performs poorly on more challenging out-of-distribution garments. Therefore, the title and abstract should be significantly weakened. | Resolved | Title; abstract; §I ¶5 |
| 17 | Separate architecture, reward, data and heuristics | The manuscript still overstates the contribution in several places. The authors should more clearly separate what is due to the model architecture, the reward design, the data collection strategy, and the workspace/action-sampling heuristics. | Resolved | §IV-C closing paragraph; §V |
| 18 | With H=1 the method is one-step latent reward optimisation | The paper emphasizes latent dynamics and future prediction, but the reported MPC horizon is only 1. This makes the method closer to one-step latent reward optimization than long-horizon model-based planning. The authors should either evaluate longer planning horizons or reframe the method more accurately. | Resolved | §IV-C; Figure 5(f) |
| 19 | Metrics miss hidden folds, layer order, front/back, usability | Coverage and Max-IoU are reasonable top-down image metrics, but they do not fully capture hidden folds, layer ordering, front/back orientation, garment topology, or downstream usability. The reported failure modes, such as folds underneath, untwisting failures, and near-success disturbance, suggest that the learned reward is still not sufficiently reliable. | **Partial** | §IV-D limitations; no new metric |
| 20 | Add a stopping criterion or a no-op action | A practical flattening system should know when to stop or avoid undoing progress. The current system appears unstable near the goal. A no-op action, stopping criterion, terminal value model, or explicit penalty for disturbing near-flat states would be important additions. | **Partial** | §V ¶5 states the gap; not implemented |
| 21 | Strongest baselines not evaluated in the real world | The strongest baselines are not evaluated in the real-world setting. The absence of real-world comparisons with MEDOR or a strong goal-conditioned Diffusion Policy weakens the claim that LaGarNet has superior real-world deployment advantages. | **Partial** | §IV-D justifies the exclusions; still not evaluated |
| 22 | Present results more conservatively | The results should be presented more conservatively. The real-world performance is promising but clearly not human-level. The OOD garment results and near-flat failures should be emphasized as limitations rather than treated as minor issues. | Resolved | Title, abstract, §IV-D, Table 6(d) |
| 23 | Discuss runtime more critically | The runtime should also be discussed more critically. Although LaGarNet is faster than MEDOR, the full real-world execution time per step is still high, which limits practical deployment. | Resolved | §IV-D closing paragraph; Table I |
| 24 | "Human-level" is an overclaim against the Human Policy | The term 'human-level' is an overclaim. In Table VI, LaGarNet performs substantially worse than the Human Policy. Therefore, I do not think the results can be described as human-level. | Resolved | Same as #16 |
| 25 | Hybrid-primitive complexity claim is unquantified | In the Related Work section, the paper claims that 'the incorporation of these hybrid strategies introduces additional complexity to the action space for data collection and control itself.' However, the paper does not quantitatively analyze the trade-off between such complexity and performance, nor does it compare against these methods. | Resolved | §II ¶2: the unsupported complexity claim is removed and replaced by an explicit scope statement |
| 26 | Clarify how GC-RSSM differs from prior goal-conditioned RSSMs | GC-RSSM can be viewed as adding goal conditioning to an existing RSSM framework, while the coverage-alignment reward is a linear combination of coverage improvement and Max-IoU improvement. However, the paper does not sufficiently clarify the essential differences between the proposed method and related prior methods. | **Partial** | §II-D; contribution 1; §II-B. The distinction is drawn, but in one sentence rather than the explicit GC-Dreamer/GCP contrast the comment invites — see comment 26 below |
| 27 | Reliance on a pre-captured goal image | The method depends on the target state of the garment, which is a relatively strong assumption in practical scenarios. In general, the system would not have access to a pre-captured flattened goal image. The paper should discuss how performance would be affected when using a canonical goal or a goal image generated by a generative model, which would make the method more practically useful. | Resolved | §IV preamble (the goal is canonical, captured once per garment, augmented in training); §IV-D limitation (generated goals out of scope) |
| 28 | Table I lists MPC horizon 1 but the paper claims long-horizon | Table I states that LaGarNet uses an MPC horizon of 1. However, the experimental conclusions claim that LaGarNet has advantages for long-horizon dynamics and tasks. The paper lacks comparison experiments with different MPC horizons. | Resolved | Same as #4 and #18 |

---

## 4. Comment by comment

### 1. Novelty versus system integration
> Many components are adaptations or combinations of existing ideas: RSSM/PlaNet-style latent dynamics, goal conditioning, CEM action selection, reward shaping based on coverage/alignment, Diffusion Policy-assisted data collection, and workspace masking. These pieces may be useful together, but the paper should more carefully distinguish between algorithmic novelty and system integration.

**Resolved.** The paper now says plainly that it is a system-integration contribution. §I: "LaGarNet
integrates these three parts into one system to answer our central research question: can latent
world models that are built for continuous control still function in quasi-static primitive
setups…". §V opens: "We present an integrate world model system that combines goal conditioning,
recurrent state-space model, a linear shaping reward, generalisable data collection and
mask-constrained CEM planning." The genuinely new pieces — GC-RSSM, the coverage-alignment reward
and the data-collection procedure — are stated as such in the contributions.

### 2. Cognitive-psychology subsection
> Section III-A, which motivates the method through cognitive psychology, is weakly connected to the actual algorithm. I recommend removing this section or reducing it to a short motivating paragraph. The space would be better used to clarify the actual test-time planning procedure, data-generation protocol, and evaluation metrics.

**Resolved.** The subsection is deleted outright. The reclaimed space went exactly where the
reviewer asked: the CEM recipe in §III-B, the full test-time procedure in Appendix A.4, and the
metric definitions at the head of §IV.

### 3. Typographical and metadata errors
> The manuscript contains many typographical and grammatical errors that should be corrected before publication. Examples include 'In this paper we shot,' 'probablisitic,' 'continous,' 'oringal,' 'readuces,' 'server self-occlusion,' and 'all model.' There are also reference and metadata errors, such as 'WTO' where the cited organization appears to be WHO.

**Resolved.** All seven named typos are gone, and the bibliography metadata is fixed.


### 4. Horizon ablation
> The paper should either add an ablation over MPC horizons, e.g., (H=1,2,3,5), reporting both performance and runtime, or explicitly justify why (H=1) is chosen. Possible explanations might include compounding model error, the dense step-wise reward, quasi-static PnP dynamics, or computational constraints. Without such evidence or explanation, claims about multi-step planning or long-horizon reasoning should be weakened.

**Resolved.** Figure 5(f) sweeps H ∈ {1, 2, 3, 4, 5}. Success falls monotonically — 73.3 %, 63.3 %,
53.3 %, 36.7 %, 30.0 % — while mean planning time rises from 4.17 s at H = 1 to 5.33 s at H = 5
(4.53, 4.78 and 5.06 s at H = 2, 3, 4). §IV-C gives the
reviewer's own explanation: one primitive causes one large deformation, so prediction error
compounds along multi-step rollouts and lookahead costs more than it buys. The appendix adds a
second reason: mask rejection constrains only the first action of a plan, because the cloth state
after the first pick-and-place is unknown.

### 5. MEDOR comparison
> The authors should either evaluate MEDOR and LaGarNet under the same garment geometry, physics, camera, initial states, and success criteria, or present the MEDOR comparison as contextual rather than a direct state-of-the-art match.

**Resolved.** MEDOR runs on the same garments, physics, initial states and success criteria as
every other baseline. The one deliberate difference is the input framing, and it is now stated and
justified: MEDOR fits its mesh against a garment template and is sensitive to how much of the frame
the garment fills, so we crop the observation passed to it until the garment-to-image ratio matches
the distribution it was tuned on. The §IV-A2 text states this, and the Figure 3(a) caption records that MEDOR "is evaluated on
cropped observations". The simulation-configuration table has been deleted and its parameters
inlined in §IV-A1, so there is no longer a separate MEDOR column implying a different setup.

### 6. Additional metrics
> The authors should additionally report final-step metrics, first-hit success, and a success-maintenance metric, for example whether the garment remains successful for one or more subsequent actions. Area-under-curve metrics would also help characterize sample efficiency and stability across the full rollout.

**Partially resolved.**
- *Final-step:* Figure 3(a) now pairs the cumulative **Max** with the **Last** value at each of 5,
  10, 20 and 30 steps, for every metric and method.
- *First-hit:* the cumulative SR is by definition first-hit — the fraction of trials succeeding at
  any step up to N — and both captions now say so explicitly.
- *Success maintenance:* quantified in §V ¶5 (76.7 % of simulated longsleeve trials reach success
  within 20 steps, but only 13.3 % hold it at the final step) and in Figure 6(b) for the real world
  (4/40 trajectories cannot stabilise on a success state; 168/298 near-success steps are followed
  by a drop).
- *Area under the curve:* **not reported.** See §5 below.
- The real-world table carries no final-step column. The caption explains why: each cell averages
  only ten physical trials, so a single final-step mean would be noisy, and Figure 6(b) measures
  end-state behaviour more directly.

### 7. Table III caption and the 20-versus-30-step discrepancy
> Table III reports blocks for 5, 10, 20, and 30 steps, but the caption does not clearly state whether SR means success exactly at that step or success achieved at least once within the first (N) actions. Based on the metric definition, the intended interpretation seems to be cumulative performance up to the horizon, but this should be explicitly stated. The paper should also clarify why dataset trajectories are described as up to 20 steps while Table III reports 30-step evaluation.

**Resolved.** Both captions now state the rule. Figure 3(a): "Each horizon $N \in \{5, 10, 20, 30\}$
spans two columns: *Max* (the highest value achieved within $N$ steps, representing cumulative
success for SR) and *Last* (the exact value at step $N$)". Table II carries the equivalent sentence
for the cumulative reading, and states that it reports no final-step column (see comment 6).

On the horizons, §IV now explains the choice: a human operator comfortably flattens every garment
type well within 20 actions, which is "already sufficient to show the difference between LaGarNet
and the baselines", so a longer headline horizon would add evaluation time without changing the
ranking. Simulated episodes run on
to 30 steps because that costs almost nothing in an automated simulator and shows how each policy
behaves once the easy gains are exhausted. Real-world trials stop at 20 because a physical step
costs about 25 s plus a manual reset.

### 8. Parameter-matched comparison
> Table IV reports that LaGarNet has fewer parameters than the Diffusion Policy baseline and MEDOR. This is useful, but it is a static comparison, not a controlled model-size study. If the authors want to claim architectural superiority or parameter efficiency more broadly, they should include a parameter-matched comparison or a small/medium/large model-size ablation for LaGarNet and the main neural baselines.

**Not planned — we decline this study.** The paper does not claim architectural superiority from
parameter count. Table I is a deployment-cost comparison, and the efficiency claim rests on
inference time (6 s versus MEDOR's 103 s), which follows from planning in a compact latent space
rather than over a reconstructed mesh. MEDOR's cost is dominated by test-time mesh fitting, not by
network size, so parameter matching would not change the comparison. See §5 for the wording change
that makes this position defensible.

### 9. Planning-horizon ablation
> The ablations examine latent dynamics, reward design, data size, and data composition, but they do not test the planning horizon. Given that the central distinction of the paper is the use of a learned world model for action selection, this is a major omission. A horizon ablation would also help determine whether the learned latent dynamics are actually useful for multi-step prediction/control or mainly useful as a one-step action scorer.

**Resolved.** The planning ablation is now the sixth ablation axis (Figure 5(f)) and also includes
an **unconstrained** variant at H = 1 that removes mask rejection. That variant collapses to 0.0 %
success and 51.0 final IoU, which answers the underlying question directly: the constrained action
space matters far more than lookahead depth. §IV-C states this conclusion rather than hiding it.

### 10. Test-time procedure
> The paper should more clearly specify the exact test-time procedure: how CEM initializes and updates the action distribution, how many elite samples are used, how workspace and cloth-mask constraints are imposed during sampling, whether action candidates are sampled in normalized image coordinates or physical coordinates, and how the selected pixel-space action is converted into robot motion. The current description gives the high-level idea but is not sufficient for reproduction.

**Resolved**, at two levels of detail.

§III-B gives the recipe: a diagonal Gaussian initialised at μ = 0, σ = 1; K = 100 iterations; N =
5000 candidates per iteration; rejection without resampling of any candidate violating the masks; a
pick pixel inside both the cloth mask and the workspace mask and a place pixel inside the workspace
mask; refitting μ and σ on the best 10 % (N_e = 500 elites); execution of the clipped mean.
Sampling and scoring happen in normalised image coordinates, and only the executed action becomes
a metric robot pose.

Appendix A.4 completes it: the pixel conversion p = (a+1)·H/2; the fact that N_e is a fixed fraction
of the drawn population rather than of the survivors, so every survivor becomes an elite when fewer
than N_e pass; early exit when an iteration leaves no survivors; one-step rollout of the
goal-conditioned prior for scoring; the restriction of mask rejection to the first action of a plan;
the sources of the cloth mask in simulation and reality; and the full pixel-to-pose chain — window
pixel → full frame via the ROI crop → camera ray intersected with the table plane using the
hand-eye transform → two end-effector poses executed with the guarded descent and grasp procedure.

### 11. Workspace radii
> Figure 9 lists (r_n, rf = 0.7m, 0.2m), which appears inverted because the far radius should normally be larger than the near radius, and the notation should be r_f. The authors should check this parameter table carefully.

**Resolved.** Figure 8(c) now reads `$r_n, r_f$ = 0.2 m, 0.7 m` — the values are in the right order
and the subscript is typeset correctly. The surrounding text in Appendix D uses the same notation
consistently.

### 12. Table density
> Table III is dense and hard to parse. The authors should explicitly state that the rows correspond to evaluation horizons and clarify whether the metrics are cumulative maxima or exact-step values. If max-over-trajectory metrics are retained, final-step metrics should be placed next to them.

**Resolved.** The simulation table is now a heatmap, Figure 3(a), in which rows are methods and
columns pair Max with Last at each horizon, so the two are literally adjacent and colour carries
the magnitude. The caption states the row and column semantics explicitly. The real-world table
keeps its tabular form — ten trials per cell do not warrant a heatmap — but its caption now states
that every block is an evaluation horizon and every value cumulative over it.

### 13. Citation metadata
> There are several apparent citation-numbering or metadata inconsistencies. For example, the text refers to VCD and MEDOR in ways that do not always match the listed reference numbers. Before publication, the authors should systematically verify that every citation number, author list, title, venue, and year is correct.

**Resolved.**  The specific VCD problem the reviewer spotted is fixed: VCD was cited
as `lin2020softgym` in two captions and is now `lin2022learning`. Nine further entries were
corrected (years, venues, page ranges, author lists) and nine were cross-checked against the
published record. BibTeX now runs with zero warnings.

### 14. Video
> If the authors intend to provide a video, it would be useful to include uncut rollouts showing both successes and failures, especially near-success disturbance cases, one-step action selection behavior, and real-robot timing. A video would be particularly important because the paper's main claims concern real-world garment flattening quality and deployment.

**Partially resolved.** Performance videos have been submitted and a narrated slide video is built
(`slides/lagarnet-talk.mp4`), and the manuscript now tells the reader the material exists. What is
actually in `main.tex` is one sentence, at the end of §I after the contributions:

> "We submit a supplementary document covering the full architecture, hyperparameters, planning
> procedure and real-robot setup, together with video material showing real-world rollouts of
> LaGarNet flattening the four garment types."

**Still outstanding, two items.**

1. There is **no first-page `\thanks` footnote** announcing the supplementary material. IEEE house
   style wants one, and it is the first place an editor looks. Suggested text: *"This paper has
   supplementary downloadable material provided by the authors. The material consists of several
   videos of real-world garment-flattening rollouts with LaGarNet."*
2. The footage on hand is **10× sped-up highlight rollouts**, not what the comment asked for. The
   reviewer specifically pre-empted a highlight reel: they asked for uncut rollouts at true speed,
   including failures and near-success disturbances. Until that footage is cut, the §I sentence
   above must stay as it is — do **not** promote it to the stronger wording:

> "Supplementary video material accompanies this paper, showing uncut real-world rollouts of
> LaGarNet flattening the four garment types at true speed, including failure cases and the
> near-success disturbances analysed in Section V."

A re-cut from the raw captures is cheaper than arguing the point.

### 15. Limited component novelty
> The novelty of the individual components is limited. RSSM-style latent dynamics, goal conditioning, CEM planning, diffusion policy data collection, and coverage-based rewards are all natural extensions of existing methods. The paper's main effort is system integration rather than a clearly new algorithmic principle.

**Resolved**, together with comment 1. The paper agrees with the reviewer and says so. §II-B on the
reward is explicit that the linear form is simple by design and that the novelty lies in the two
terms and in computing both from segmentation masks alone, so one reward serves both simulation and
the real world.

### 16. "Human-level" in the title and abstract
> The claim of 'human-level' garment flattening is also not well supported. In real-world experiments, human operators still substantially outperform LaGarNet in success rate and final metrics. The method also performs poorly on more challenging out-of-distribution garments. Therefore, the title and abstract should be significantly weakened.

**Resolved.** The title changed from "LaGarNet: Human-Level Pick-and-Place Garment Flattening with
Goal-Conditioned Recurrent State-Space Models" to "LaGarNet: Latent World Models for Quasi-Static
Pick-and-Place Garment Flattening". The abstract was rewritten and makes no human comparison at
all; it now claims only parity with mesh-based methods at five times fewer parameters. §I says
LaGarNet "is also competitive with human operators under the same single-arm constraint, although
humans succeed more often at long horizons".

### 17. Attribution of the gains
> The manuscript still overstates the contribution in several places. The authors should more clearly separate what is due to the model architecture, the reward design, the data collection strategy, and the workspace/action-sampling heuristics.

**Resolved.** §IV-C closes with an explicit ranking: "the ablations rank the sources of LaGarNet's
performance. Mask-constrained action sampling matters most… The reward comes next… The architecture
contributes least on its own: RSSM-GC-I nearly matches GC-RSSM on the task metrics, and separates
from it mainly in prior reconstruction quality. We therefore credit the combination rather than the
latent model alone." §V repeats the point, including the observation that constraining the action
space matters more than the latent architecture.

### 18. One-step latent reward optimisation
> The paper emphasizes latent dynamics and future prediction, but the reported MPC horizon is only 1. This makes the method closer to one-step latent reward optimization than long-horizon model-based planning. The authors should either evaluate longer planning horizons or reframe the method more accurately.

**Resolved.** Longer horizons are now evaluated and shown to be worse (Figure 5(f)), and the paper
reframes accordingly: §IV-C concludes "This justifies T_f = 1", and the framing throughout is
quasi-static one-step planning rather than long-horizon reasoning. The title itself now says
"Quasi-Static".

### 19. What the metrics capture
> Coverage and Max-IoU are reasonable top-down image metrics, but they do not fully capture hidden folds, layer ordering, front/back orientation, garment topology, or downstream usability. The reported failure modes, such as folds underneath, untwisting failures, and near-success disturbance, suggest that the learned reward is still not sufficiently reliable.

**Partially resolved.** §IV-D states the limitation in the reviewer's own terms: LaGarNet "fails to
isolate and untwist narrow sections such as sleeves and trouser legs"; distinguishing a garment's
front from its back "was also not a task criterion, so LaGarNet cannot perform this semantic
orientation"; and the reward predictor "is least reliable once the garment lies flat". Figure 6(a)
and Table 6(b) enumerate the failure modes.

**Still outstanding:** no new metric that captures hidden folds, layer ordering or downstream
usability has been added. The paper acknowledges the blind spot rather than measuring it.

### 20. Knowing when to stop
> A practical flattening system should know when to stop or avoid undoing progress. The current system appears unstable near the goal. A no-op action, stopping criterion, terminal value model, or explicit penalty for disturbing near-flat states would be important additions.

**Partially resolved.** §V ¶5 concedes the point directly: "LaGarNet still does not know when to
stop", quantifies the instability (76.7 % versus 13.3 %; 168 of 298 near-success steps end in a
drop), diagnoses the cause ("the reward predictor is least reliable once the garment lies flat, so
the planner keeps acting when it should stop"), notes that human operators stop as soon as the
garment looks flat, and names a learned stopping criterion as the fix.

**Still outstanding:** no stopping criterion, terminal value model or near-flat penalty is
implemented or evaluated. This is stated as future work.

### 21. Real-world baselines
> The strongest baselines are not evaluated in the real-world setting. The absence of real-world comparisons with MEDOR or a strong goal-conditioned Diffusion Policy weakens the claim that LaGarNet has superior real-world deployment advantages.

**Partially resolved.** §IV-D now gives the reasons rather than leaving the gap unexplained: "MEDOR
is excluded because of its prohibitive execution time and the difficulty of tuning its camera
setup, and Diffusion Policy because adapting its action inference to our workspace constraints is
non-trivial." The second reason is developed in §V, which explains that enforcing workspace masks
during action denoising is difficult and that a Diffusion Policy trained without those constraints
does not transfer — which is itself part of the argument for planning.

**Still outstanding:** neither baseline is actually run on the robot, so the real-world comparison
remains against PlaNet-ClothPick and the human policy only.

### 22. Conservative presentation
> The results should be presented more conservatively. The real-world performance is promising but clearly not human-level. The OOD garment results and near-flat failures should be emphasized as limitations rather than treated as minor issues.

**Resolved.** The human-level claim is gone from the title, abstract and introduction. §IV-D is a
dedicated "Limitations and Failure Cases" subsection covering the sim-to-real gap, multi-task
capacity (the unified policy underfits longsleeves and trousers while overfitting skirts and
dresses), the trousers-as-sleeves confusion, disruption of already-flattened garments, and the
out-of-distribution dress of Table 6(d). §V ¶5 leads with what the system cannot do.

### 23. Runtime
> The runtime should also be discussed more critically. Although LaGarNet is faster than MEDOR, the full real-world execution time per step is still high, which limits practical deployment.

**Resolved.** §IV-D ends with the breakdown and its consequence: perception 8.19 s, action
inference 7.05 s, robot execution 9.20 s, so a single manipulation step takes about 24.5 s and a
20-step episode over eight minutes. The paper states that "time consumption is a critical
limitation for deployment in fast-paced garment factories" and that scaling to industrial or
domestic use requires optimising both the latent-dynamics inference and the perception pipeline.

### 24. "Human-level" as an overclaim
> The term 'human-level' is an overclaim. In Table VI, LaGarNet performs substantially worse than the Human Policy. Therefore, I do not think the results can be described as human-level.

**Resolved**, as for comment 16. The real-world table (now Table II) is unchanged and still shows
the gap plainly — 27/40 versus 38/40 successes at 20 steps — and the surrounding text no longer
claims parity. The caption states that LaGarNet "still falls short of human-level performance".

### 25. Hybrid-primitive complexity
> In the Related Work section, the paper claims that 'the incorporation of these hybrid strategies introduces additional complexity to the action space for data collection and control itself.' However, the paper does not quantitatively analyze the trade-off between such complexity and performance, nor does it compare against these methods.

**Resolved by removing the claim.** The reviewer is right that the trade-off was asserted without
evidence, so the assertion is gone. §II ¶2 no longer says hybrids "complicate both data collection
and control"; it now states the scope decision instead:

> "Hybrids of these three primitives are highly effective for diverse garment manipulation. We
> adopt a single-arm pick-and-place setup, which isolates our research question from the additional
> design choices and challenges introduced by a hybrid action space with a dual-arm setup. We
> therefore neither evaluate nor compare our method against more complex setups, leaving hybrid
> primitives to future work."

**Response-letter line.** Hybrid primitives are outside the scope of this paper. The question we
set out to answer is whether latent world models function in a complex quasi-static setup, and
holding the action space fixed at single-arm PnP is what makes that question answerable — a hybrid
action space would confound the result with primitive selection. We therefore withdraw the
unsupported complexity claim rather than defend it with a study the paper does not need.

### 26. GC-RSSM versus prior goal-conditioned RSSMs
> GC-RSSM can be viewed as adding goal conditioning to an existing RSSM framework, while the coverage-alignment reward is a linear combination of coverage improvement and Max-IoU improvement. However, the paper does not sufficiently clarify the essential differences between the proposed method and related prior methods.

**Partially resolved.** §II-D describes both prior methods and then draws the distinction in a single
sentence: "Duan et al. present the goal-directed exploration algorithm MUN and devise GC-Dreamer as a
baseline, but condition the goal on its actor-critic networks instead of the world model. Unlike the
above methods, we condition the RSSM's latent inference on the goal directly; Section IV-C isolates
this choice against other goal-conditioning variants." Contribution 1 repeats the delta, and the
ablation backs it empirically: RSSM-GC-I and RSSM-GC-IO concatenate the goal image to the input
without changing the dynamics, and are separated from GC-RSSM by prior reconstruction quality.

On the reward, §II-B states that the linear form is simple by design and that the novelty is in
the two terms and in computing both from segmentation masks alone, so a single reward serves both
simulation and the real world.

### 27. Reliance on a goal image
> The method depends on the target state of the garment, which is a relatively strong assumption in practical scenarios. In general, the system would not have access to a pre-captured flattened goal image. The paper should discuss how performance would be affected when using a canonical goal or a goal image generated by a generative model, which would make the method more practically useful.

**Resolved by clarification.** The premise was ours to correct: LaGarNet already uses a *canonical*
goal, not a per-trial capture, and the manuscript never said so. §IV now states the protocol
explicitly:

> "This goal is canonical rather than trial-specific: one image per garment, captured once and
> reused across every trial."

with the independent goal augmentation stated in the same paragraph ("due to the independent
augmentation on the goal and current images during training").

The §IV-D limitations paragraph adds the honest boundary:

> "LaGarNet also assumes access to a canonical flattened image of each garment; this is a one-off
> capture rather than a per-trial demonstration, and our goal augmentation during training keeps
> the policy from depending on its exact pose, but removing the assumption altogether — for
> instance by synthesising the goal with a generative model — is outside the scope of this study."

**Response-letter line.** We use a canonical goal image, obtained once per garment and reused
throughout, and we augment it independently during training precisely so the policy does not
depend on a pixel-exact target — the assumption is therefore weaker than the comment supposes.
Goal images produced by a generative model are a separate research question and out of scope here.

### 28. Horizon 1 versus long-horizon claims
> Table I states that LaGarNet uses an MPC horizon of 1. However, the experimental conclusions claim that LaGarNet has advantages for long-horizon dynamics and tasks. The paper lacks comparison experiments with different MPC horizons.

**Resolved**, by the same horizon ablation as comments 4, 9 and 18. The conclusions no longer claim
a long-horizon advantage; they claim that a recurrent latent world model still learns useful
dynamics in the quasi-static regime, and that greedy one-step planning over that model is the right
choice for this action primitive.

---

## 5. Still outstanding — what a reviewer might object to again

### Comment 8 — parameter-matched study, declined
**Position: a model-size sweep is not necessary, and the cover letter should say so directly.**
The paper does not claim architectural superiority from parameter count. Table I is a
deployment-cost comparison — it exists to show that latent planning is cheap enough to run on a
robot, not that 7.11 M parameters beat 37.5 M by virtue of being fewer. The efficiency claim is
carried by inference time (6 s versus 103 s) and is a property of planning in a compact latent
space rather than over a reconstructed mesh, which no amount of parameter matching would change:
MEDOR's cost is dominated by test-time mesh fitting, not by network size.

**Risk if we simply decline:** the abstract and §I both say "five times fewer parameters", which
does read as an efficiency claim about the architecture. Leaving that phrasing while refusing the
study is the one thing likely to draw the comment again.

**Cheapest defensible fix:** add to the Table I caption that it
compares the configurations as published and is not a controlled model-size study, and let the
inference-time argument carry the efficiency claim. Optionally soften "five times fewer parameters"
to a joint statement about parameters *and* inference cost, which is what the evidence supports.

### Comment 14 — the video is pointed at, but two things are still open
§I announces the supplementary material, so the reader is told it exists. Two gaps remain: there is
no first-page `\thanks` footnote (IEEE house style, and the first place an editor looks), and the
footage is a 10x sped-up highlight cut rather than the uncut real-speed rollouts with failures that
the reviewer explicitly pre-empted. §4, comment 14 holds both the footnote text and the stronger §I
sentence, ready to drop in once the re-cut exists.

### Comments 25 and 27 — closed by scoping, not by new experiments
Both are now addressed in the manuscript, but neither is answered with an experiment, so the
cover letter should say so plainly rather than let the reviewer discover it.

- **#25:** the unsupported complexity claim is withdrawn and §II ¶2 states that hybrid primitives
  are outside the paper's scope. We do not compare against hybrid methods and no longer claim
  anything about their cost.
- **#27:** the premise is corrected — the goal is canonical, captured once per garment and reused,
  and independently augmented during training — and the §IV-D limitation records that generated
  goal images are out of scope. No canonical-versus-generated experiment was run.

### Partial items worth one more pass
- **#6 (AUC)** — every other requested metric is now reported. If the rollout data is available,
  an AUC column would close the comment completely.
- **#19, #20, #21** — all three are acknowledged in the text as limitations or future work rather
  than measured. That is a defensible position, but I think it should be stated as a deliberate scope
  decision in the cover letter, not left for the new reviewer to infer.
