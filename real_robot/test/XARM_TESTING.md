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
- [ ] Arm assignment is right: the larger-pixel-x pick goes to the **left** arm.
      If the script warns about this, the camera roll assumption is wrong.

### 5c. Pick-and-fling — ⚠️ last, and expect a failure first

```bash
python real_robot/test/test_xarm_primitives.py --primitive fling --case all --dry-run
```

- [ ] `invalid` case aborts before moving (0 waypoints checked)
- [ ] `basic` / `wide`: **currently 2 unreachable waypoints each** (see below)
- [ ] Only once the dry run is clean: `--case basic --execute`

**Known failure (offline, at the measured 0.66 m separation).** *Ten* waypoints
fail, for two independent reasons:

```
!! left movel[0] xyz=(+0.105, +0.000, +0.300)  XY radius 0.105 m inside the 0.120 m base keepout
!! left movel[1] xyz=(+0.105, -0.450, +0.300)  3D distance 0.551 m exceeds the ~0.44 m reach
```

1. **Stretch too wide.** The skill stretches to `±width/2` about the base midpoint,
   so each TCP ends up `|S/2 − width/2|` from **its own** base. With `S = 0.66` and
   `STRETCH_MAX_WIDTH = 0.45` that is `0.105 m` — *inside* the 0.12 m base keepout.
   Counter-intuitively, a **narrower** stretch pushes the TCPs further out: the
   usable range is `width ≤ S − 2 × 0.12 = 0.42 m`. `STRETCH_MAX_WIDTH ≈ 0.36`
   gives `r = 0.15 m`.
2. **Swing too long.** `SWING_STROKE = 0.45` exceeds the entire Lite 6 reach.
   At `r = 0.15` and `HANG_HEIGHT = 0.25`, a stroke of `0.25` lands at `0.384 m` —
   comfortable.

Candidate set: `STRETCH_MAX_WIDTH 0.36`, `HANG_HEIGHT 0.25`, `SWING_STROKE 0.25`.
All three trade away fling energy, so decide against the real `--reach` numbers from
step 4. **This is a deliberate open decision, not an oversight.**

- [ ] Retune `STRETCH_MAX_WIDTH` / `HANG_HEIGHT` / `SWING_STROKE`
- [ ] Re-run the dry run until clean, then execute

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
  is a stand-in only.
- Fling dynamics tuning (stroke, swing angle, speeds) against real fabric.
- Aligning controller firmware v2.2.2 / v2.3.0.
- If xArm results are ever reported, `paper/example.tex` needs the real cell
  geometry in place of the UR description.
