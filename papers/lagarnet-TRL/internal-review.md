# LaGarNet — internal review and how the T-RL manuscript resolves it

This is a working document for the T-RL resubmission, separate from `comments.md` (which tracks the
28 external reviewer comments). The 30 comments below come from the co-author's internal read of
the manuscript and are numbered in the order they appeared. Every revision made in response is
highlighted **in blue** in `main.tex` via the `\rev{}` macro; setting `\showrevfalse` in the
preamble produces an identical but unhighlighted camera-ready copy.

**Status counts: 25 resolved, 4 partially resolved, 1 blocked.**

---

## 1. The review was read against an older PDF

Two comments had already been fixed before the review arrived, and reference numbers have moved
since:

| Reviewer's observation | Current state |
|---|---|
| `"mobility-impaired populations [3], [?]"` — missing reference | The dangling `\cite{disability2011world}` was dropped in the previous redundancy pass; the sentence now cites `WHOaged2024` alone |
| `"LaGarNet instead couples a goal-conditioned…"` | Already rewritten to `"LaGarNet instead integrates…"`; this pass only adds **novel** |
| Reference numbers `[21]`, `[3]` etc. | The bibliography is split into a 45-entry main list `[1]`–`[45]` and a 9-entry appendix-only list `[46]`–`[54]`; adding SAC and Klein shifted numbers again. See `README-bib.md` |

## 2. Structural changes in this pass

| Change | Detail |
|---|---|
| Author block | Halid: York **and** St Andrews; John: York **and** Loughborough; Kasim: St Andrews. Three `\thanks` markers |
| New reference | `klein1983predressing` — Klein, *Pre-Dressing Skills*, Communication Skill Builders, 1983. Bibliography regenerated to 45 + 9 |
| Removed | Table II's `\cellcolor` shading (×4) and the caption's `\colorbox` highlights (×4) |
| Removed | The stray `\noindent` opening §II |
| Renamed | $\gamma_{klo}, \mathcal{L}_{klo} \to \gamma_{KLo}, \mathcal{L}_{KLo}$; $\mathcal{R}_{du} \to \mathcal{R}_{dU}$ |
| Restored | The MPC planning-time measurement in §IV-C, dropped in the previous pass (see §5) |

Six paragraphs reflowed into a 1–3 word last line after these edits and were trimmed; all paragraph
runt lines are clear.

---

## 3. Resolution table

| # | Gist | Reviewer comment | Status | Where in T-RL |
|---|---|---|---|---|
| 1 | Affiliation should be York and St Andrews | Your affiliation should be both York and St Andrews, I think, since some of your work was at St Andrews and you will be based here again from September (after publication). | Resolved | Author block; new `\thanks{$^{3}$}` for Loughborough |
| 2 | Is `\noindent` after a section heading against IEEE rules? | Are there any rules about using \noindent on first line of a section? It looks strange when it's indented, but maybe it's IEEE rules... | Resolved | There **is** a rule and it favours the indent — see comment 2 below. The one `\noindent` in §II is removed |
| 3 | Figure 1 caption names the wrong garment | Figure 1 caption says "yellow dress are out-of-distribution". I only see a white dress and a yellow skirt? | Resolved | Figure 1 caption |
| 4 | Missing reference on p1 | "mobility-impaired populations [3], [?]" -- missing reference | Resolved | Already fixed before the review; §I ¶1 |
| 5 | "is particularly a foundational" | "is particularly a foundational" -> "in particular is a foundational" | Resolved | §I ¶1 |
| 6 | "physical reasoning" is a loaded phrase | I'm not sure what is meant by "physical reasoning" here. It is clearly not the gripper because human policy deals fine with that, it is more about the model itself. I guess what you mean is that it is reasoning about the physics of the cloth, but the word "physical" to me implies the physical manipulation by the robot, so it's a bit loaded. | Resolved | §I ¶3: "reasoning about cloth deformation" |
| 7 | Highlight GC-RSSM as a contribution | "LaGarNet instead couples a goal-conditioned" -> "a novel goal-conditioned...". This is one of the contributions so highlight it. | Resolved | §I ¶4 |
| 8 | "Transferred zero-shot" is not grammatical | Maybe consider "In a zero-shot setting..."? | Resolved | §I ¶5 |
| 9 | "and Hoque et al. [21] propose" | -> "while Hoque et al. [21] propose..." | Resolved | §II-A ("whilst", for consistency with the manuscript) |
| 10 | "In contrast, our mesh-free" | -> "In this paper, we show that our mesh-free", or "we show that our mesh-free..." | Resolved | §II-A |
| 11 | "reduces them" | -> "reduces these biases" | Resolved | §II-B |
| 12 | "and transferred zero-shot to real garment" does not flow | Do you mean "and is transferred in a zero-shot way to real garment..."? It's just the phrasing that does not flow with the rest of the sentence. | Resolved | §III opening: "transferred to real garments without any real-world fine-tuning" |
| 13 | "giving the GC-RSSM as" | -> "defining the GC-RSSM as" | Resolved | §III-A |
| 14 | Eq. 4 optimises over θ but θ does not appear | I'd recommend changing p to p_theta (there is space) because theta parameterises p(.) You do this for r in Eq12. | Resolved | Equation 4. Equations 5–6 still use bare `p`/`q` — see §6 |
| 15 | Eq. 5 needs a trailing comma | In first reading, I thought that p(.) multiplies q(.) before noticing the equation labels. | Resolved | Equation 5 |
| 16 | "free nat" wording | Maybe simply remove the second "nat", like this: "... threshold c=1 that clips" | Resolved | §III-A, exactly as suggested |
| 17 | What does "klo" stand for in Eq. 10? | Is it "KL+overshooting"? If so, I would capitalise KL everywhere to make it more obvious. | Resolved | $\gamma_{KLo}$, $\mathcal{L}_{KLo}$ in §III-A and Table III; now glossed as "the KL and reward overshooting losses" |
| 18 | T_f = 1 is a hook for a reviewer | "we set the prior future prediction horizon to Tf = 1, as shown to be optimal via empirical study" -- this looks like a potential weakness someone might latch onto (as reviewers did in the previous iteration). We need to downplay the long-horizon aspect because of this. | **Partial** | §III-B now gives the mechanism and forward-references §IV-C. The property itself is not removable |
| 19 | Eq. 13: why "R_du"? | Why "R_du" instead of "R_dU" or "R_dIoU"? | Resolved | §III-C: $\mathcal{R}_{dU}$, matching $\mathcal{R}_{dC}$ |
| 20 | The 20-step cap may disadvantage the baselines | "which already separate LaGarNet from the baselines" -> "which are already sufficient to show the difference between LaGarNet and the baselines". Someone might ask whether this disadvantages the baselines, because some of them could (in theory) beat LaGarNet's best performance but take more than 20 steps to do it. Maybe a note to say that in experiments the baselines plateaued after a certain limit is reached? | **Partial** | §IV opening reworded as suggested. The plateau note is **not** added — see comment 20 below |
| 21 | "if two checkpoints both exceed 1.9/2.0" is unclear | what is being measured here? | Resolved | §IV opening: "score above 1.9 out of the maximum 2.0" |
| 22 | "driven by their superior Max IoU" | what does it mean that humans have a "Max IoU" that is superior to something else? Maybe rephrase it in more natural language rather than the language used to define computational rewards? | Resolved | §IV-A3: "because they align the garment more precisely with its flattened shape" |
| 23 | All-Garment > Task-Specific warrants analysis | This is a really interesting and warrants highlighting and analysis. If this is analysed later, I would reference that section here. | **Partial** | §IV-A3 now points to Figure 3(b) and §V. There is no dedicated analysis to point at — see comment 23 below |
| 24 | Figure 3(a) encodes information by colour | This will not work when printing BW (as many people do for reading papers), but more importantly, it presents problems for people with disabilities, especially disabilities relating to colour. Simply numbering the methods would be easier to parse and will occupy the same amount of space. | **Blocked** | Agreed, but the figure cannot be re-rendered — see comment 24 below |
| 25 | Figure 3(a) caption still unclear | "columns pair, at each of 5, 10, 20 and 30 steps..." -- I'm afraid that this is a bit unclear still. | Resolved | Figure 3(a) caption rewritten around "Each horizon N contributes two adjacent columns" |
| 26 | The horizon sweep needs motivation | "The planning ablation sweeps H ≡ Tf ∈ {1, 2, 3, 5}" -- this might need a bit of motivation, you are comparing the case where Tf = 4? | Resolved | §IV-C baselines: "covering short horizons densely and probing longer lookahead at H = 5". H = 4 is not evaluated |
| 27 | Table II's yellow/grey shading is unexplained | Are these meant as comparisons to previously published mesh baselines? I would think that mentioning these in text would be easier to understand, leaving the table for reporting the results you measured in experiment only. | Resolved | All shading removed; the caption now says the VCD/MEDOR figures come from those papers and the table reports only our own trials |
| 28 | Figure 6(b)'s "as-if-manipulating-other-object" is not glossed | I can see it later in the main text (p10) but you do not describe it in the caption (and you do describe other cases). | Resolved | Figure 6 caption |
| 29 | Figure 6(d)'s 0/10 success rate is hard to defend | what it seems to show is that the method does not work on out-of-sample pieces. This could be used to question the ability to do zero-shot transfer like you claimed earlier. The NC, NU and IoU results are interesting, but 0/10 SR will lead someone to simply say "it does not work". | **Partial** | Figure 6(d)'s sub-caption now reports the 77.2 NC alongside the 0/10 — see comment 29 below |
| 30 | "challenging even for young children" needs a citation | citation? | Resolved | §IV-D, `\cite{klein1983predressing}`. **Not** the T-RO citation — see comment 30 below |

---

## 4. Comment by comment (the ones that need more than a line)

### 2. `\noindent` after a section heading
> Are there any rules about using \noindent on first line of a section? It looks strange when it's indented, but maybe it's IEEE rules...

**Resolved — and it is IEEE rules.** `IEEEtran.cls:5472` defines `\section` through `\@startsection`
with a **positive** before-skip (`3.0ex plus 1.5ex minus 1.5ex`). LaTeX's `\@startsection` only sets
`\@afterindentfalse` when that skip is negative, so IEEEtran deliberately leaves
`\@afterindenttrue`: the first paragraph of every section is indented by design, which is why
IEEE Transactions papers look that way. §II was the only section opting out, so the `\noindent`
is removed rather than propagated.

### 14. θ in Equation 4
> Eq4: argmax is over theta, but theta is not shown in the term being optimised.

**Resolved.** Equation 4 is now `\argmax_\theta \ln p_\theta(...)`. Equations 5 and 6 have the same
property — their `p` and `q` are also parameterised — and are left bare for now; say the word and
they can be subscripted to match.

### 18. T_f = 1 as a potential weakness
> ...this looks like a potential weakness someone might latch onto (as reviewers did in the previous iteration). We need to downplay the long-horizon aspect because of this.

**Partial.** The bare assertion "as shown to be optimal via empirical study" is replaced by a
mechanism and a forward reference: *"we set the prior future prediction horizon to T_f = 1:
Section IV-C shows that longer horizons degrade performance, because a single primitive produces
one large deformation and prediction error compounds across a multi-step rollout."* That turns an
unexplained hyperparameter into a stated finding backed by Figure 5(f), which is the strongest
available defence. Beyond that, H = 1 is a property of the method, not a defect to be fixed —
external comments 18 and 28 in `comments.md` cover the same ground.

### 20. Do the baselines plateau before 20 steps?
> Maybe a note to say that in experiments the baselines plateaued after a certain limit is reached?

**Partial.** The wording change is made. The plateau note is deliberately **not** added, because the
paper's own data contradicts it: in Figure 3(a) the task-specific Diffusion Policy's success rate
climbs from 43.3 % at 20 steps to 80.0 % at 30. Writing that the baselines have plateaued would be a
false claim in a figure the reader can check on the same page. The 20-step cap is defensible on
evaluation cost — each real trial is roughly 8 minutes of robot time — not on saturation, and §IV
now says only that 20 actions suffice to *show the difference*.

### 23. All-Garment versus Task-Specific
> This is a really interesting and warrants highlighting and analysis. If this is analysed later, I would reference that section here.

**Partial.** §IV-A3 now signposts Figure 3(b) and §V, but there is no dedicated analysis anywhere in
the paper to reference. Explaining *why* morphological diversity helps would need a new experiment —
per-garment probing of the shared latent, or a controlled data-mixture sweep at fixed volume — which
is beyond a revision. Flagged here so it is a deliberate omission rather than an oversight.

### 24. Figure 3(a)'s colour-only encoding
> Fig3a is problematic because you are using colours to encode information... Simply numbering the methods would be easier to parse and will occupy the same amount of space.

**Blocked, and the criticism is correct.** Figure 3(a) identifies each of the 8 method rows by a
coloured dot and each of the 8 column pairs by a coloured square, both resolved only through a
legend. That fails in greyscale printing and for readers with colour-vision deficiency, and numbered
or text row labels would indeed cost no space.

It cannot be fixed right now: `plot_lagarnet_metrics_heatmap` in
`analysis/lagarnet/flattening_comparison_for_lagarnet_new.ipynb` (cell 4) reads
`/media/halid/T7/garment_folding_data/lagarnet_data`, and the T7 drive is not mounted. The same
blocker applies to Figure 5(a)–(d). The fix is roughly ten lines in that cell — replace the dot/square
legend with text row labels and `Max`/`Last` column headers, keeping colour only for the heatmap
cells themselves — and should be run the next time the drive is available.

⚠ Related: `analysis/lagarnet/ablation_{latent_bar,reward_bar,data_size_line}.png` are newer than
the copies in `plots/` but **empty** — every bar renders at zero because they were regenerated
without the data drive. Do not copy them into `plots/`.

### 29. The 0/10 out-of-distribution success rate
> ...0/10 SR will lead someone to simply say "it does not work". This is better in the main text, but it still feels a bit risky.

**Partial.** Figure 6(d)'s sub-caption now reads: *"the policy still recovers most of the garment's
area (77.2 NC), but never reaches the strict joint NC/IoU success criterion."* That puts the
partial-progress reading in the same sentence as the zero, so a skimming reader meets both at once.
The result itself stands — no wording makes 0/10 look like success, and softening it would be worse
than the risk, particularly given external comments 16, 22 and 24 which pushed the paper towards
*more* conservative reporting.

### 30. A citation for "challenging even for young children"
> p11: "a spatial task known to be challenging even for young children." -- citation?

**Resolved, but not with the T-RO citation.** The T-RO manuscript cited `paoletti2012pink` here
(`lagarnet-TRO/main.tex:1452`). That reference is Jo B. Paoletti, *Pink and Blue: Telling the Boys
from the Girls in America* (Indiana University Press, 2012) — a cultural history of gendered
children's clothing in the United States. It says nothing about spatial orientation or the
front/back discrimination of garments, so restoring it would have been a mis-citation; it was
already dropped in the previous reference-pruning pass for that reason.

The claim now cites Marsha Dunn Klein, *Pre-Dressing Skills: Skill Starters for Self-Help
Development* (Communication Skill Builders, Tucson AZ, 1983), a standard reference on the normal
developmental sequence of dressing and undressing, including orienting a garment. It resolves as
[43]. A search for a peer-reviewed alternative turned up Hayton et al., *Frontiers in Education* 4:149
(2019) on dressing-skill development, but that paper covers fastenings rather than garment
orientation and does not support this specific claim.

---

## 5. Changes made in this pass that the review did not ask for

**The §IV-C horizon numbers are now verified against the run logs, not read off the figure.**
`/home/halid/lagarnet_data` holds all five horizon evaluations and the stored notebook output
reproduces them exactly: SR 73.3 / 63.3 / 53.3 / 30.0 at H = 1, 2, 3, 5, unconstrained SR 0.0 and
final IoU 51.0 (the text said 50.9). The caveat carried since the previous pass is discharged.

**The planning-time measurement is restored** with correct values — mean per action rises from
4.17 s at H = 1 to 5.33 s at H = 5. External reviewer comment 4 explicitly asked for *performance
and runtime*, so dropping this in the previous pass was a regression against an already-resolved
comment. ⚠ `comments.md` §4 comment 4 still quotes the superseded figures (50.0 / 40.0 / 26.7 /
20.0 % and 4.10 → 5.18 s) and needs updating.

**The horizon ablation's protocol is corrected.** §IV-C claimed the 20-step protocol; the stored run
used `max_steps=30`, so the text now says the planning ablation evaluates over 30 action steps.
(It also used `iou_thresh=0.79` against the stated 0.80 success criterion — almost certainly a
float-tolerance guard, but text and run differ by a point. Not changed.)

**A "surpass the state of the art" claim is qualified.** §IV-D said LaGarNet's NC and NI surpass the
values reported by VCD and MEDOR. That holds on the aggregate row but not like-for-like: the
longsleeve 20-step NI is 71.1 against VCD's reported 79.2 on real T-shirts, and the trousers 10-step
NI is 57.2 against MEDOR's 64. The sentence now reads "averaged over all four garments… which each
report on a single garment type."

## 6. Open items

- Equations 5 and 6 still use unparameterised `p` and `q` (comment 14).
- Figure 3(a) and Figure 5(a)–(d) await the T7 drive (comment 24).
- `comments.md` comment 4 quotes superseded horizon numbers (§5 above).
- Symbol collisions carried over from earlier passes: `$H$` is both the MPC horizon and the image
  height; `$K$` is the CEM iteration count, the diffusion step count, and the number of SAM
  candidate masks.

## 7. Verification

```bash
cd papers/lagarnet-TRL
latexmk -pdf -bibtex -synctex=1 -interaction=nonstopmode main.tex   # exit 0
grep -i "undefined\|multiply defined" main.log                      # empty
grep -c Overfull main.log                                           # 1, unchanged
pdfinfo main.pdf | grep Pages                                       # 17
```

Last run: exit 0, no undefined citations, 1 Overfull box, 17 pages, all paragraph runt lines clear,
`klein1983predressing` → [43], main list `[1]`–`[45]`, appendix list `[46]`–`[54]`.
