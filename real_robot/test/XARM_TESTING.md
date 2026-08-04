# Dual xArm Lite 6 — hardware test runbook

A checklist for bringing the two-arm cell up and testing each primitive on real
hardware. Work top to bottom: every step assumes the ones above it passed.

**Cell:** two UFACTORY xArm Lite 6, mounted on the two **long (120 cm) edges** of an
80 × 120 cm table, facing each other across the short axis. Measured base-to-base
centre distance **0.66 m**, so each base is inset ~7 cm from its 120 cm edge and the
arms are centred across the 80 cm width. Along the length they are **not** centred:
the arm line is **0.52 m from the front 80 cm edge** and 0.68 m from the back.
Both arms carry a **UFACTORY Gripper for Lite 6** (pneumatic — needs an air supply).

**Table coordinates** used throughout = the **LEFT arm's base frame**:
`x` runs from 0 at the left base across the short axis to 0.66 m at the right base;
`y` runs along the arm line (the table's 120 cm axis) with the **front of the setup
at −y**; `z` is up, table at `z = table_z`. The table edges are therefore
`x ∈ [−0.07, +0.73]`, `y ∈ [−0.52, +0.68]`.

---

## Simulation — no robots needed

A PyBullet model of the cell, with the top-down camera and a click UI. The
**unmodified** primitives run against it, so this is the place to try things before
the arms are switched on.

```bash
python real_robot/sim/fetch_lite6_assets.py          # once: ~3.1 MB into assets/
python real_robot/sim/run_sim_ui.py --primitive fling        # click 2 pick points
python real_robot/sim/run_sim_ui.py --primitive dual-pnp     # click 4
python real_robot/sim/run_sim_ui.py --primitive single-pnp --arm left
python real_robot/sim/run_sim_ui.py --headless --auto --primitive fling   # self-check
```

`--delay 0.002` makes the swing watchable; `--seed N` moves the garment; `--loop`
keeps going. The click helpers are the same `human_utils` ones the real-robot human
policies use.

**What is real and what is not:**

| | |
|---|---|
| Kinematics | **real** — UFACTORY's own URDF; its FK reproduces poses measured on your hardware to **2–4 mm** |
| Camera | **real projection**, built from the same intrinsic `point_on_table_base` inverts (agrees with the analytic model to 0.9 px) |
| Cloth | **not** physics — a drawn catenary between the grippers |
| IK | PyBullet's solver, **not** the xArm controller's |
| Joint effort | not simulated — reports "no signal", so the primitives take their hardware fallback |

> The sim validates sequencing, framing and gross reachability. The controller's own
> `get_inverse_kinematics` (what `IKGuardedArm` uses) stays the authority before real
> motion, and `test_xarm_primitives.py --offline` stays the fast gate.

The headless run asserts every commanded pose was reached, no joint limit was
exceeded and the arms never touched — checks the offline reach-*sphere* test cannot
make, because it has no kinematic chain.

After a run the UI redraws the frame with the executed TCP paths on it. Those are the
paths' **shadow on the table**, not where the camera would see them: with the camera
1.0 m up, a gripper at the 0.25 m fling height is magnified ~40 %, which moved the
swing by up to 52 px in a 330 px window and pushed the wind-up off the top edge. The
shadow puts each grasp exactly on the pixel you clicked, at the cost of not showing
height. Raising the camera shrinks the gap between the two.

### Which way up the frame is

The frame is laid out the way you would draw the cell standing in front of it: the
**left arm on the left**, the **right arm on the right**, the **front edge at the top**.

Those are one choice, not two. For a camera looking straight down the image frame is
right-handed with the view direction as camera +z, so "u runs toward base +x" forces
"v runs toward base −y" — which is why the front (0.52 m) edge is at **base +y** and
the back (0.68 m) edge at −y. Flip one of those without the other and you get a frame
that is upside down and virtual walls that protect the wrong half of the table.

Nothing depends on it any more: the dual-arm skills assign arms by position on the
**table** (`sort_pairs_by_table_x`), not by pixel column, so a camera bolted on rotated
gives a view you dislike rather than a swapped grasp. It used to matter — the skills
sorted on pixel x, and a roll silently handed each arm the other one's target.

- [ ] Once the camera is mounted, check the `[xarm-camera] frame: …` line printed at
      scene init against what you see. `-- as intended` means the mount matches the
      convention above; anything else tells you which way it actually came out.

### What the camera hands you: a square window between the arms

Every frame — in the sim and on the robots — is a **square crop of the table centred
on the midpoint of the two arm bases**, not the raw camera image. The scene crops it
and shifts its own intrinsic's principal point to match, so a pixel in that window
still inverts to the right point on the table and no call site needs to know.

| knob | where | default |
|---|---|---|
| `XARM_CROP_SIZE` | `xarm_constants.py` | `0.66` m — the base separation, i.e. the square spanned between the arms |

**This is the one to tune on hardware.** It is a length on the *table* in metres, so
it keeps meaning the same patch when the camera height or the lens changes. At the
default it resolves to 330 px in the sim (f = 500 px at 1.0 m) and ~396 px on a
RealSense colour stream at 1.5 m — both upsampled to the arena's 512. If you enlarge
it much past ~1.1 m the arms' own bases come into frame; if you make it too big to
fit, `crop_window` says so loudly rather than quietly sliding the window off-centre.

### The photo pose: each arm swings to its own left

Before every capture the arms go `home()` → `out_scene()`, and `out_scene` is now
**home with joint 1 rotated +90°**, derived per arm rather than written out. Joint 1
turns about the base z axis, so the TCP keeps home's height and radius: if the taught
home clears the table, so does this. The right base is yawed 180°, so the left arm
tucks toward the table's back and the right toward its front — they separate.

Measured in the sim, this cuts the arms' footprint inside the crop by **41 %**
(6.3 % → 3.7 % of the frame). Set `XARM_PHOTO_YAW` to change the angle; whatever you
set, only joint 1 may move (`test_xarm_walls_offline.py` enforces that).

- [ ] After teaching (step 4), watch one `out_scene()` on each arm before trusting it

---

## 0. Every session

```bash
cd ~/Projects/bimanual_garment_folding
source ./setup.sh xarm      # conda env + PYTHONPATH + wired NIC on the arm subnet
```

`xarm` also exports `XARM_LEFT_IP` (192.168.1.155) and `XARM_RIGHT_IP` (192.168.1.170),
which every script below picks up as its default. It is a no-op (and asks for no
sudo) if the NIC is already on the robot subnet.

- [ ] `ping -c2 192.168.1.155` and `ping -c2 192.168.1.170` both answer

---

## 1. Arms healthy — no motion

```bash
python real_robot/test/test_xarm_lite6_bringup.py --info-only --arm both
```

- [ ] Both arms print **`OK: no errors, arm is ready to move.`**

> **Known outstanding:** the RIGHT arm last reported `error_code = 2, state = 4` —
> *emergency IO of the control box triggered*. Nothing below will run until it is
> cleared: release the e-stop, and check the **EI terminals** on the right control
> box (if there is no external e-stop, those terminals need the shorting link
> fitted). The driver already runs `clean_error()` on connect, so an error that
> survives is physically asserted. While `state = 4` the reported joint/pose values
> are stale — do not trust them.

Also noted: firmware is **v2.2.2 (left)** vs **v2.3.0 (right)**. Not blocking, but
worth aligning eventually.

---

## 2. Driver conversion check — small motions

The one real correctness risk in the whole port: `XArmLite6` converts metres+rotvec
to mm+RPY assuming scipy `'xyz'` Euler order. A wrong order still runs — it just
commands wrong orientations at the cloth.

```bash
python real_robot/test/test_xarm_lite6_bringup.py --arm left     # then --arm right
```

- [ ] **Stage 1** (re-command the measured pose): the arm **does not visibly move**,
      and rotation error < 1°
- [ ] **Stage 2** (±3 cm jogs in +Z, +X, +Y): measured delta matches commanded within
      a few mm; note which physical direction each base axis points
- [ ] **Stage 3** (±5° joint-1 jog): returns to the start joints

If stage 1 fails, fix `rotvec_to_rpy` / `rpy_to_rotvec` in
`real_robot/robot/xarm_lite6.py` before going any further.

Once both arms pass individually:

```bash
python real_robot/test/test_xarm_lite6_bringup.py --arm both --dual
```

- [ ] Both arms jog simultaneously and stay clear of each other

---

## 3. Grippers — no arm motion

Safe to run with the arms parked. **Air supply must be on and connected.**

```bash
python real_robot/test/test_xarm_gripper.py --arm both
```

Stages, in order:

1. **Report** — firmware (needs ≥ 1.10.0 for the Lite6 gripper API) and tool-IO state
2. **Raw tool-IO** — drives DO0/DO1 directly, bypassing the driver. Separates "the
   code is wrong" from "there is no air / the valve is not wired". If nothing moves
   here, no driver change will help.
3. **Driver cycle** — times `open_gripper()` / `close_gripper()` so the 0.6 s
   `sleep_time` in the driver can be tuned to the real actuation time
4. **HOLD test** — ⭐ the important one
5. **Fabric** — can it grip and keep gripping garment cloth

### ⭐ Record the HOLD verdict

`XArmLite6.close_gripper()` does *close → sleep → `stop_lite6_gripper()`*, and
`stop` drives **both DO lines low**. Whether the fingers keep holding then depends
on the valve:

| Result | Meaning | Action |
|---|---|---|
| **HOLDS** | double-solenoid valve latches | nothing to do — the driver is fine |
| **RELEASES** | spring-return valve vents | **every primitive grasp silently drops the garment.** Stop and report — the driver needs a hold path that leaves DO1 driven |

- [ ] Left verdict: `HOLDS` / `RELEASES` → ______
- [ ] Right verdict: `HOLDS` / `RELEASES` → ______
- [ ] Fingers fully open and close every cycle (if not, raise `sleep_time` or air pressure)
- [ ] Grips a single layer of the garment fabric and survives a gentle tug

---

## 4. Teach the cell — measure the missing constants

`XARM_TABLE_Z`, `XARM_GRIPPER_OFFSET`, `XARM_HOME_JOINT` and
`XARM_WORKSPACE_RADIUS` in `real_robot/utils/xarm_constants.py` are still
**unverified guesses**. The primitives descend to the table and call `home()`, so
these must be measured first.

### ⚠️ First: confirm both controllers report in the same frame

**Run step 2's `--info-only` on both arms and check `tcp_offset` and `world_offset`
are zero before teaching anything.** The teach script now refuses to measure
otherwise (override: `--allow-offsets`), because every value it records is a
coordinate read back from the controller — a non-zero offset does not add noise, it
measures a different frame, and the result looks exactly like data.

This is not hypothetical. On 2026-07-31 the right controller had a leftover
`tcp_offset` of `[-288.7, 0, +238.2] mm` — a phantom 374 mm tool — and the left had
none. The symptoms, all from that one setting:

| | left | right |
|---|---|---|
| flange z with fingertips on the table | +0.0869 m | **−0.1530 m** |
| home TCP, for joints matching within 5.5° | `[0.220, 0.032, 0.466]` | **`[−0.069, −0.040, 0.225]`** |
| z spread over 3 table touches | 0.1 mm | **29.9 mm** |
| IK reach at grasp height | 0.050–0.430 m | **0.050–0.125 m** |

Two arms of the same model at the same joint angles **must** report the same TCP —
the base frame travels with the base, so mounting height and orientation cannot
change it. A constant offset across unrelated poses is a frame transform, not a
measurement. Clear it with:

```bash
python real_robot/test/test_xarm_lite6_bringup.py --arm right --clear-tcp-offset
```

which zeroes the tool frame and calls `save_conf()` so it does not return on the
next reboot. Then re-read with `--info-only` to confirm.

### Then teach, one arm at a time

```bash
python real_robot/test/test_xarm_teach.py --arm left  --all
python real_robot/test/test_xarm_teach.py --arm right --all
```

Measurements are stored **per arm** in `calibration/xarm-cell.yaml` under
`arms.<side>`, and the constants have `*_BY_SIDE` dicts to match. They used to be
cell-wide, where teaching the second arm silently overwrote the first's numbers.

`--all` = `--table-z --home --reach`. Free-drive uses `set_mode(2)`, which makes
the arm **back-drivable — take its weight before each prompt.** Position mode is
always restored, including on Ctrl-C.

> [Virtual walls](#virtual-walls) are currently **off** (`XARM_GEOMETRY_VERIFIED`
> is `False`), so nothing here will be refused. Once they are on, hand-guiding past
> a wall trips controller error 35 — the script clears it and continues, but add
> `--no-walls` for this step, since the floor wall sits at `XARM_TABLE_Z`, the very
> thing `--table-z` is measuring.

- [ ] **`--table-z`** — rest the *closed fingertips* flat on the table at 3 spots.
      Gives `XARM_GRIPPER_OFFSET`. Watch the z-spread warning: >5 mm across spots
      means the table is not level in the base frame and a fixed-height grasp will
      miss on one side. Pass `--table-height` if the table is not at base z = 0.
- [ ] **`--home`** — hand-guide to a ready pose: TCP well above the table, pointing
      down, clear of the other arm. Gives `XARM_HOME_JOINT`.
- [ ] **`--reach`** — *no motion*; sweeps the controller's IK outward at the grasp,
      lift and hang heights. Gives `XARM_WORKSPACE_RADIUS`, and prints whether the
      fling's stretch target is commandable.

Then the base geometry (needs both arms). Put **one mark** on the table near the
midline; the 0.14 m workspace overlap means both arms can touch it. (Use
`--mark-dx` for two marks a tape-measured distance apart if you prefer.)

```bash
python real_robot/test/test_xarm_teach.py --arm both --separation --marks 3
```

Both arms touch the **same** marks, so no tape measure and no mounting assumption
is involved. Three marks, spread along the arm line rather than clustered, are fit
for yaw + translation (2D Kabsch). **Two is the minimum** — one shared point gives
2 equations for 3 unknowns and cannot determine the yaw at all, which is precisely
why it was an assumption until now.

- [ ] Measured separation is within ~2 cm of 0.66 m
- [ ] Fit residual < 10 mm (else the marks were touched imprecisely — re-measure)
- [ ] Height difference < 1 cm (else one `XARM_TABLE_Z` cannot serve both arms)
- [ ] **Note the measured yaw** — if it differs from the assumed 180° the script
      says so loudly, and every right-arm target has been landing in the wrong place

### ⚠️ Then paste the printed constants into `xarm_constants.py`

The primitives do `from ...xarm_constants import XARM_TABLE_Z`, binding the value
at **import** time — so measurements sitting in `calibration/xarm-cell.yaml` have
no effect until they are pasted in. The primitive script cross-checks the two and
refuses to run if they disagree.

- [ ] `XARM_GRIPPER_OFFSET_BY_SIDE`, `XARM_TABLE_Z_BY_SIDE`,
      `XARM_HOME_JOINT_BY_SIDE`, `XARM_WORKSPACE_RADIUS_BY_SIDE` updated **for both
      sides** (the script prints the exact line, keyed by arm)
- [ ] `XARM_BASE_SEPARATION`, `XARM_BASE_YAW` updated
- [ ] `XARM_GEOMETRY_VERIFIED = True` — switches the virtual walls on
- [ ] The two arms' gripper offsets agree to within a centimetre. They are at the
      same table, so they must — `test_xarm_primitives.py` refuses to run if they
      disagree by more, since that means a frame error rather than two measurements

---

## 5. Primitives

`test_xarm_primitives.py` drives the **shipped** skill classes unmodified. There is
no camera and no hand-eye calibration yet, so it fabricates a synthetic top-down
camera from the measured geometry: targets are written in **table metres** and
converted to the pixels the primitives expect. Every waypoint is checked against
the controller's own inverse kinematics before it is sent.

Three modes, least to most dangerous:

| Flag | Hardware | Motion | Use |
|---|---|---|---|
| `--offline` | none | none | pure geometry check, works at a desk |
| `--dry-run` (default) | connected | none | real IK checks against the controller |
| `--execute` | connected | **yes** | the real thing |

`--offline` and `--dry-run` both override `--execute`, so an accidental combination
never moves an arm.

### 5a. Single-arm pick-and-place

```bash
python real_robot/test/test_xarm_primitives.py --primitive single-pnp --case all --dry-run
python real_robot/test/test_xarm_primitives.py --primitive single-pnp --case basic --execute --arm left
```

Cases: `basic`, `rotated` (π/4 grasp rotation), `route-around` (transit crosses the
base keepout, so an extra waypoint should appear at ≈ `(0.120, 0.000)`).

- [ ] Dry run: 0 unreachable for all 3 cases
- [ ] `--execute --arm left`, then `--arm right`
- [ ] The gripper actually picks the cloth up and puts it down where expected

### 5b. Dual-arm pick-and-place

```bash
python real_robot/test/test_xarm_primitives.py --primitive dual-pnp --case all --dry-run
python real_robot/test/test_xarm_primitives.py --primitive dual-pnp --case left-only  --execute
python real_robot/test/test_xarm_primitives.py --primitive dual-pnp --case right-only --execute
python real_robot/test/test_xarm_primitives.py --primitive dual-pnp --case both       --execute
python real_robot/test/test_xarm_primitives.py --primitive dual-pnp --case collision  --execute
```

- [ ] `both` → prints `active -> left: True, right: True` and runs the **simultaneous**
      path, each arm reaching `(0.250, -0.120)` in its own base frame
- [ ] `collision` (picks 8 cm apart, inside the 0.12 m threshold) → prints
      `Collision predicted; executing sequentially` and runs the **sequential** path
- [ ] Arm assignment is right: the pick nearer the **left base on the table** goes to
      the left arm. The skill compares base-frame x, not pixel x, so this holds however
      the camera is rolled. If the script warns about it, the *case* is mis-named.

### 5c. Pick-and-fling — ⚠️ last, and bring it up stage by stage

```bash
python real_robot/test/test_xarm_primitives.py --primitive fling --case all --dry-run
```

- [ ] `invalid` case aborts before moving (0 waypoints checked)
- [ ] `basic` / `wide`: all waypoints reachable (46 and 32 respectively)
- [ ] Only once the dry run is clean: `--case basic --execute`

The skill mirrors the UR `PickAndFlingSkill` stage for stage —
approach → probe → grasp → lift/centre → **stretch** → **shake** → re-read →
swing → drag → **tension release** → open. The Lite 6 has no F/T sensor, so the
UR's three force-mode stages are gated on **joint effort**
(`XArmLite6.get_joint_effort`, firmware ≥ 1.9.0) instead:

| UR | here |
|---|---|
| `move_until_contact`, 5 N | descend in 5 mm steps to the **calibrated** grasp height, stopping early on an effort rise. Bounded below, so it can only ever stop *above* the table |
| force-mode stretch, 6 N | step outward until the width cap, a timeout, or an effort rise |
| force-mode release, −2 N | a 3 cm inward position move before opening |

> ⚠️ The effort signal is weak — this is a 0.61 kg-payload arm and a garment weighs
> almost nothing. Every stage keeps a hard geometric cap, and behaves correctly if
> the effort never fires. Tune `XARM_EFFORT_THRESHOLD` (per arm) from the deltas the
> stretch prints. If it proves unusable, the width cap still bounds the motion.

Bring it up one stage at a time — each can be switched off:

```bash
--skip-probe     # descend straight to the calibrated grasp height
--skip-shake     # no vertical shake after the stretch
--skip-release   # open without the inward tension release
```

**The fling constants are derived, not chosen.** `get_base_fling_poses` builds the
swing in a frame centred between the bases, so each gripper sits at
`x = (S − width)/2` from **its own** base and the wind-up waypoint is at
`(x, −stroke, hang)`. Two constraints follow:

```
(1) base keepout:  (S − width)/2  >=  r_min          ->  width <= S − 2·r_min = 0.42 m
(2) reach:         x² + stroke² + hang²  <=  r_max²
```

At `S = 0.66` and the measured `(0.12, 0.41)`: **width 0.36, hang 0.25, stroke 0.25**
puts the swing at `r_3d = 0.363 m`, ~11% inside the limit. That is about half the UR
cell's fling (width 0.65, hang 0.35, stroke 0.65) — a UR5e reaches ~0.85 m, so this
is geometry, not timidity.

`test_xarm_walls_offline.py` asserts both constraints, so if you change
`XARM_BASE_SEPARATION` or the measured reach it fails on a laptop rather than at the
arm. Re-derive the constants in `xarm_constants.py` rather than editing them by feel.

- [ ] `--dry-run` clean, then `--execute --skip-shake --skip-release`
- [ ] Add the shake, then the release
- [ ] Tune `XARM_EFFORT_THRESHOLD` and `XARM_FLING_SPEED`/`ACC` against real fabric

---

## Virtual walls

A box the TCP may never leave: **the four table edges, the tabletop as a floor, and
everything above it allowed** up to a ceiling well past the arm's reach.

### ⚠️ Currently OFF by default

`XARM_GEOMETRY_VERIFIED` in `xarm_constants.py` is `False`, so the walls are **not
enforced yet**, and `XArmLite6` says so at connect.

The walls are placed using `XARM_BASE_YAW` — how the right base is rotated relative
to the left — which was *assumed* to be 180°. Hardware contradicts it: with the
right gripper physically over the table, that arm reported its TCP at
`x = −0.239 m` in its own base frame, i.e. the table is at **negative** x for it.
A box placed with the wrong frame does not protect the table; it refuses legitimate
moves. So enforcement waits until the frame is measured.

- [ ] Run `test_xarm_teach.py --arm both --separation` (step 4), paste the measured
      `XARM_BASE_YAW` / `XARM_BASE_SEPARATION`, then set
      `XARM_GEOMETRY_VERIFIED = True` — that single flag switches the walls on

### The box, once enabled

Defined once in table coordinates (`XARM_WALLS`) and transformed into each arm's
own base frame by `real_robot/utils/xarm_walls.py`:

| axis | wall | why |
|---|---|---|
| x | `−0.02 … +0.68 m` | the 80 cm table edges, inset 5 cm |
| y | `−0.47 … +0.63 m` | the 120 cm table edges, inset 5 cm (asymmetric — the arm line is 0.52 m from the front) |
| z | `table_z … table_z + 0.50 m` | **floor at the tabletop**; everything above it is allowed |

The yaw mirrors both horizontal axes, so the two arms program **different** y limits:
left `[680, −20, 630, −470, 500, 0]` mm, right `[680, −20, 470, −630, 500, 0]` mm.
The x limits coincide only because the bases are equidistant from their edges. Most
of these walls sit beyond the ~0.44 m reach and will never trigger — the ones that
bite are the **z floor** and the x wall near the base.

**Two layers, both driven by the same box:**

1. **Controller fence** — `set_reduced_tcp_boundary` + `set_fence_mode(True)`,
   programmed at connect and read back to confirm it took. Nothing bypasses it, not
   even free-drive or UFACTORY Studio. A violation halts the arm with **error 35**.
2. **Driver check** — every waypoint, Cartesian *and* joint (via forward kinematics,
   so `home()` is covered). Out-of-box moves are **refused, never clamped**: nothing
   is sent, `movel` returns `False`, and the log names the wall and the overshoot.
   A trajectory with one bad waypoint sends **none** of it.

`set_reduced_mode` is deliberately left off — it would also cap speed and throttle
the fling.

### ⚠️ The fence lives in the controller and persists

`set_fence_mode` writes to the **controller**, not to this process. It survives
disconnect, process exit and a power cycle. A boundary programmed by a run three
days ago is still armed today, and nothing in the log will mention it.

That is why the driver **reconciles the fence on every connect** rather than only
programming it when walls are enabled: `walls=False` (and `walls='auto'` while the
geometry is unverified) actively calls `set_fence_mode(False)` and says so.

**If you see controller error 35 ("safety boundary limit"):** the TCP is outside
*some* boundary — not necessarily one this run programmed. While the TCP sits
outside an armed box the controller refuses **every** move, in any direction, even
one heading back inside. `_ok()` now prints the live boundary, the current TCP and
which axis is out. To clear it: connect with walls off, or call
`driver.disable_walls()`.

*(This is exactly what stopped the right arm on 2026-07-31: a fence from an earlier
walls-on run, with the TCP 219 mm outside it in x.)*

### Checking and overriding

```bash
python real_robot/utils/xarm_walls.py             # print the boxes, no hardware
python real_robot/test/test_xarm_walls_offline.py # 13 offline checks, no hardware
```

Stage 0 of the bring-up prints the **persistent** controller settings — fence,
boundary, reduced mode, `world_offset`, `tcp_offset`. Check them before trusting
any coordinate:

- [ ] `safety_boundary_is_on` matches what the driver reported (`1` with the
      expected boundary when walls are on, `0` when off)
- [ ] `reduced_mode_is_on = 0` — otherwise speeds are capped and the fling is throttled
- [ ] `world_offset` and `tcp_offset` are **zero** — a non-zero one means
      `get_position` is not reporting base-frame coordinates, so `XARM_TABLE_Z`, the
      walls and the hand-eye calibration would all be measured in the wrong frame

`--no-walls` on `test_xarm_teach.py` and `test_xarm_lite6_bringup.py`; in code
`XArmLite6(..., walls=False)` or `driver.disable_walls()` (both layers together).
`walls=True` forces them on even before the geometry is verified.

> **Limits:** bounds the **TCP only** — elbows can still leave the box — and does
> **not** prevent arm-vs-arm collision. That stays `check_trajectories_close` plus
> collision detection. The floor is only as good as `XARM_TABLE_Z`.

---

## Safety notes

- Every script stops the arms (`set_state(4)`) and disconnects on Ctrl-C or any error.
- Virtual walls are on by default — see above.
- Nothing calls `home()` / `out_scene()` until step 4 has measured a real home pose.
- `test_xarm_primitives.py` defaults to 0.10 m/s — deliberately slow. Raise with
  `--speed` only after a primitive has run cleanly.
- Keep a hand on the e-stop for anything with `--execute`.

## Reach reality check

At the conservative `XARM_WORKSPACE_RADIUS = (0.12, 0.40)` and the measured 0.66 m
separation, the two arms **genuinely overlap** over `x ∈ [0.26, 0.40]` — a band
0.14 m deep about the midline. Step 4's `--reach` sweep measures the real radius and
may widen it further. Dual-arm grasps on one garment still work best with the two
grasp points on opposite halves, but a handover or a shared-point touch at the
midline is now possible — which is what makes the one-mark `--separation`
measurement work.

`python real_robot/test/xarm_test_scene.py` prints the current reach map and
verifies the synthetic camera round-trip, with no hardware at all.

## Still open after this runbook

- Real hand-eye calibration (`hand_to_eye_calib.py`) to replace the identity
  placeholders in `calibration/xarm-{left,right}-calib.yaml`; the synthetic camera
  is a stand-in only. The crop centre is a real projection through `T_left_cam`, so
  it follows the calibration; the crop *size* assumes the camera looks straight
  down, and goes slightly non-square on the table if calibration finds a tilt.
- Tune `XARM_CROP_SIZE` against the real camera, and settle its height — 1.0 m
  clips the 1.20 m table length for a RealSense colour stream, ~1.50 m does not,
  and the height also sets how many pixels the crop gets.
- Fling dynamics tuning (stroke, swing angle, speeds) against real fabric.
- Aligning controller firmware v2.2.2 / v2.3.0.
- If xArm results are ever reported, `paper/example.tex` needs the real cell
  geometry in place of the UR description.
