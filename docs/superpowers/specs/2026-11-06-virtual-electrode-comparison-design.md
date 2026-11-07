# Virtual electrode on the muscle: a fair model vs MaM sEMG comparison

Date: 2026-11-06.
Status: approved direction, not yet implemented.

## 1. Intent

Figs 2 and 3 compare MaM sEMG with our model using a whole-muscle aggregate (summed joint travel per muscle in `compare_ziva_run.py`).
We want a one-on-one comparison: the model at the exact spot under each MaM electrode against that electrode's signal, over the course of the expression.
Example: the model's orbicularis oris right under the oor_r electrode vs the MaM `Orbicularis Oris_r` channel.

The EMG is only the enabler of the chain EMG → excitation-contraction coupling → force → shortening → movement.
Our model outputs the later links.
The comparison must therefore happen at the earliest link the model has, and must never claim a unit match (µV vs mm).

We work on muscles, not skin.
The electrode sits on skin, but the quantity we read is the muscle underneath it.

## 2. Facts this design rests on

- A model landmark is one of 300 sampled vertices per muscle mesh; the 300 joints sit on them.
- The meshes are much denser: orbicularis oris has 10,240 vertices and 20,480 triangles, a closed ring (outer and inner surface, upper and lower lip), about 92 x 75 mm, so scene units are mm.
- A run stores every step's joint actions (`final_epoch_action_list.pkl`, 1,772 x 3,603 in `hpy`) but the full muscle meshes only at the final step (`deltaB_temp.pkl`, `s_next` etc.).
- Passive muscles (zygomaticus major/minor, levator labii superioris, mentalis, depressor labii inferioris, procerus, depressor supercilii, ...) exist in the Maya/Ziva scene but are not saved by runs.
- The scene also contains `skin_mesh`.
- MaM ships EMG plus 52 MediaPipe blendshape scores per frame, with no face points and no electrode coordinates.
- MaM is under a data-use agreement: all outputs stay local in gitignored `visualization/` folders.

## 3. Why a single landmark vs the electrode is not fair as-is

1. Place: an electrode sees a patch of about 1 cm; the muscle average spans about 9 cm.
2. Layer: on a ring mesh, the nearest vertices by straight-line distance can sit on the inner surface or on the other lip, which moves the opposite way.
3. Quantity: translation is the last link; a lip point also moves because zygomaticus pulls the corner.
4. Clock: solver steps are not seconds, and the step size follows our sigmoid limit schedule.

## 4. Method

### Step 1. Electrodes on our skin

- Place both electrodes of each bipolar pair on `skin_mesh` using the Fridlund placement rules already traced.
- Store them as barycentric coordinates on skin triangles.
- Later upgrade: per-participant positions from MaM electrode-session videos, which give the real placement spread.

### Step 2. Skin to muscle

- Cast a ray from each electrode along the inward skin normal.
- The first muscle mesh hit is the muscle under the electrode.
- Record hit point, depth and muscle name.
- If the hit muscle is not the channel's nominal muscle (for example the oor_r ray hits zygomaticus first), flag the channel as crosstalk-prone in our anatomy.
- Cross-check depths against the MRI.

### Step 3. Virtual landmark

- Point: store the hit point as barycentric weights on its triangle; its position at any step is those weights applied to the deformed triangle.
- Patch: all vertices of that muscle's outer surface within geodesic radius r of the hit point, weighted by vertex area x Gaussian(geodesic distance), with sigma = r / 2.
- r is set from MaM's inter-electrode distance and electrode diameter (order of 1 cm).

This replaces the n-nearest-neighbour inverse-distance average with four fixes:

- geodesic, not straight-line, distance, so it cannot jump to the inner surface or the other lip;
- restricted to one surface of one muscle;
- area-weighted kernel instead of 1/d, so it is independent of mesh density and finite at d = 0;
- radius from electrode physics instead of a count n.

Single-landmark admissibility test (per site):

- A real landmark may stand in for the virtual one if its geodesic distance to the hit point is below the placement uncertainty, and its trajectory stays within 10% RMS of peak of the patch estimate (10% is a default to revisit).
- Report which sites pass.
- Expectation: frontalis passes, orbicularis oris near the corner fails.

### Step 4. Quantities

- Primary: shortening between the two virtual landmarks of the pair, `eps(t) = 1 - |p1(t) - p2(t)| / |p1(0) - p2(0)|`.
  Pairs lie along the fibres, so this is local fibre strain under the electrode.
  Positive means shortening, negative means passive stretch.
- Supporting: translation of the patch centre (Figs 2 and 3 made local).
- Supporting: local drive, the joint travel inside the patch; zero for passive muscles; the model's enabler.

### Step 5. Common clock: expression progress

- Human: per clip, blendshape progress for the task (for happy, `mouthSmileLeft/Right`), 0 at rest, 1 at apex.
- Model: `1 - keyframe distance / initial distance`.
- Resample both onto progress 0 to 1.
- Human amplitude: `(envelope - rest) / ref`, as `mam_data` already computes.
- Model amplitude: eps divided by its own peak.
- Use single-stage runs (such as `hpy`, AU0 to AU6+12+25) for trajectories; chains contain undo stages.

## 5. Tests per channel

| Test | Question | Statistic |
| --- | --- | --- |
| Trajectory shape | Do EMG and eps rise together along progress? | Spearman rho on the progress grid; ceiling = leave-one-participant-out rho among humans; participant bootstrap CI |
| Sign | Where EMG is active, does the model shorten? | Agreement rate over (channel, task) cells |
| Recruitment class | EMG active/silent x model driven-and-shortening/not | Agreement, missing agonist, over-recruitment, agreement-silent |
| Placement sensitivity | Does the verdict flip when the electrode moves within r? | Range of the metric over the sweep |
| Optimizer dependence | Do numerical (Fig 2) and analytical (Fig 3) runs agree? | Same metrics per run; disagreement means the trajectory reflects the solver |

Expected for happy: zygomaticus is a missing agonist, which motivates adding driven muscles.
Shortening alone cannot separate active contraction from a passive muscle going slack; the drive column separates them.

## 6. Work items

1. Maya replay: re-run stored actions and export all muscle meshes, passive included, every k steps.
   Until then, only endpoints for the 4 driven muscles (rest from the keyframe, final from `deltaB_temp.pkl`).
2. Electrode-pair coordinates on `skin_mesh` and the ray cast.
3. Geodesic patch and barycentric embedding.
4. Progress alignment for both sides and a comparison script next to `compare_smile_run.py`, writing to gitignored `visualization/`.

## 7. Checks before trusting results

- Model left vs subject left: current scripts assume the run's "l" naming is the anatomical left; verify geometrically.
- MaM inter-electrode distance and electrode diameter from the dataset paper (sets r).
- Possible lead: MaM `mask/` electrode-session frames show about 10 small interior holes absent in `mask_noelec/`; they may be electrode cutouts that give per-participant electrode positions.
  Needs a visual check locally.

## 8. Limits to state

- Shortening is not activation: isometric effort and lip pressing give EMG without shortening.
- Crosstalk: bipolar sEMG picks up neighbouring muscles, especially around the mouth.
- One model head vs many participants; normalization removes scale, not anatomical differences.
- The model's trajectory is a solver path shaped by the step-limit schedule, not physiology; the progress clock reduces but does not remove this.

## 9. Open decision

- Whether shortening under the electrode pair is the headline result with point translation as the supporting view, or the reverse; to confirm with the supervisor.

## 10. Later

- Paper text (motivation, problem statement, analysis of the plots) in the latest SIGGRAPH Overleaf template.
