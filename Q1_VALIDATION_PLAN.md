# Q1 validation plan: recruitment on the MMM dataset

Q1 asks whether the simulator drives the muscles that people actually use for each pose.
This plan turns that question into steps you can run on the dataset in `MMM/MMM`.

## 1. What the downloaded data actually is

The folder holds **Mimics, Muscles, and Me (MMM)**, not the 36-person Mimics and Muscles (MaM) cohort that `VALIDATION_PLAN.html` describes.
The differences change what Q1 can claim:

| | MaM (assumed in VALIDATION_PLAN.html) | MMM (what is on disk) |
| --- | --- | --- |
| Participants | 36 | 1 (`pid` 99) |
| sEMG sessions | 2 per person | 2 (`1S`, `2S`) |
| Signal | 4096 Hz bipolar | 30 Hz RMS envelope, about 500 ms window, already frame aligned to video |
| Channels | 22 Fridlund | 22: 11 muscles x left/right |
| License | data agreement | open use, no redistribution |

What follows from that:

- Q1 becomes a **single-subject case study with two repeated sessions**. Session-to-session agreement is the only reliability check you get, and nothing here supports claims about people in general.
- The envelope is fine for "which muscle, how much" (recruitment). It is too smoothed for onset timing, so leave timing to Q4 and Q5.
- No data agreement means you can show figures and numbers in the paper. You still cannot copy the dataset into the repo. Add `MMM/` and `MMM.zip` to `.gitignore`.
- For a group-level result, keep the published Jena sEMG maps (Schumann 2021, Mueller 2022, Guntinas-Lichius 2023) as the second comparison.

## 2. A problem found in the first data check

I ran a quick pass over the `schaede` clips in both sessions.
For each channel I took the 90th percentile of the envelope per task, subtracted that channel's `Face-At-Rest` median, and divided by the channel's maximum across tasks.

Most tasks look as expected:

- `Forehead` is led by both frontalis channels.
- `Smile-Closed` is led by `Zygomaticus` on both sides in both sessions.
- `Nose-Pucker` is led by `Levator Labii Superioris`.

Two tasks look wrong:

- `Eye-Tight` is led by **`Orbicularis Oris`** at 1.00 on both sides in both sessions. `Eye-Gentle` is also led by `Orbicularis Oris_l`.
- `Lip-Pucker` has **`Orbicularis Oculi_r`** as its first or third channel.

That pattern suggests the `Orbicularis Oculi` and `Orbicularis Oris` columns are **swapped** in the released CSVs, or that crosstalk is heavier than expected.
Q1 relies on exactly these two muscles, so this has to be settled before you compute any metric (step 1 below).

The check also showed that per-channel max normalization saturates: many tasks hit 1.00 on several channels at once.
Step 2 replaces it with something more discriminating.

## 3. Mapping MMM tasks to simulator poses

The simulator's poses (from the `a3-stochastic-V1` code): `AU0_neutral`, `AU1`, `AU2`, `AU1+AU2`, `AU4`, `AU6`, `AU7`, `AU12`, `AU12L`, `AU22`, `AU25`, `AU6+12+25`.

| Simulator pose | MMM task | Expected human prime mover (MMM channel) | Pairing quality |
| --- | --- | --- | --- |
| AU0_neutral | `Face-At-Rest` | none (baseline) | exact |
| AU1+AU2 | `Forehead` | `Medialer` + `Lateraler Frontalis` | good. AU1 and AU2 separately have no MMM task |
| AU6+AU7 / AU7 | `Eye-Tight` | `Orbicularis Oculi` | good |
| AU7 (light) | `Eye-Gentle` | `Orbicularis Oculi` (palpebral part) | fair |
| AU12 | `Smile-Closed` | `Zygomaticus` | good |
| AU6+12+25 | `Smile-Open` | `Zygomaticus`, `Orbicularis Oculi` | good |
| AU12L | none | `Zygomaticus_l` | no MMM task. Use the left/right ratio of `Smile-Closed` only as a sanity check |
| AU22 | `Lip-Pucker` | `Orbicularis Oris` | fair. A pucker is closer to AU18. Compare against the artist's AU22 sculpt first |
| AU4 | none in `schaede`; `angry` frames | `Corrugator Supercilii`, `Depressor Supercilii Procerus` | weak. Emotion clips mix several AUs |
| AU25 | none | `Depressor Ang. Oris`, `Mentalis`, jaw | none |
| (no pose) | `Nose-Pucker`, `Snarl`, `Cheeks`, `Depress-Lip` | | outside the model's pose set. Use them for the channel check in step 1 |

That gives **6 strong or fair pairs** (AU0, AU1+2, AU7, AU12, AU6+12+25, AU22) and one weak pair (AU4).

## 4. What the simulator side needs

Activation is a constant 0.8 (`training/evaluator.py:1948`, `:2459`, `training/analysic_gradient/pytorch_analytical_gradient.py:217`), and only five structures are driven: the jaw, orbicularis oris, orbicularis oculi L/R and frontalis.
So Q1 cannot compare activation with EMG. It compares **recruitment vectors**: per pose, how much work the model assigns to each muscle against how much EMG each human site shows.

Define the model's recruitment vector per pose over the 11 MMM muscle sites:

- For each driven structure, use its Hill-model muscle work for the transition `AU0_neutral -> pose`, taken from the evaluator.
- Give every passive or absent muscle a value of **0**. This is a modeling statement: the solver puts no drive into those muscles. Report it as such, not as missing data.
- Also record, for a secondary analysis, the **passive deformation** of each muscle mesh (length change from the Ziva tissues). That separates "the muscle moves" from "the muscle is driven".

Model muscle to MMM channel:

| Model structure | MMM channel(s) | Status in model |
| --- | --- | --- |
| frontalis | `Medialer Frontalis`, `Lateraler Frontalis` | driven |
| orbicularis oculi L/R | `Orbicularis Oculi_l/_r` | driven |
| orbicularis oris | `Orbicularis Oris_l/_r` | driven |
| jaw | `Masseter` (opposite sign: the masseter closes the jaw) | driven. Exclude from the main metric |
| zygomaticus major | `Zygomaticus` | passive |
| levator labii superioris | `Levator Labii Superioris` | passive |
| depressor labii inferioris / DAO | `Depressor Ang. Oris` | passive. Check whether the scene has a DAO mesh at all |
| mentalis | `Mentalis` | passive |
| procerus, depressor supercilii | `Depressor Supercilii Procerus` | passive |
| corrugator | `Corrugator Supercilii` | check whether a mesh exists |

## 5. Steps

### Step 0. Loader (0.5 day)

- Write `validation/mmm.py`, which reads `MMM.csv` with `pid` as a string, loads the EMG CSV per clip, clamps it at zero and returns a frames x 22 array plus channel names.
- Pair `N` and `S` clips on `task_seq`, not on `task` (see "Known gaps" in the MMM README).
- Write a pytest test against two real clips: 22 columns, row count equal to the video frame count.

### Step 1. Settle the oculi/oris labels (1 day, blocking)

- Watch `video_noelec/99_2S/*Eye-Tight*` and `*Lip-Pucker*` next to the two envelopes.
- Correlate each envelope with MediaPipe blendshapes from `video_noelec_params/MediaPipe`: `eyeBlinkLeft/Right` and `eyeSquint*` should track orbicularis oculi, and `mouthPucker` and `mouthFunnel` should track orbicularis oris. A swap shows up as the off-diagonal correlation being the larger one.
- Check left/right too: `Smile-Closed` and `Snarl` are close to symmetric, so a left/right swap is hard to see there, but `eyeBlinkLeft` against the `_l` channel will show one.
- If the labels are swapped, fix it in the loader with a comment pointing to the evidence, and report it to the dataset authors.

### Step 2. Human recruitment vectors (2 days)

For each paired task and session:

1. Find the hold phase: the frames where the summed MediaPipe activity of the task's blendshapes is above 50% of its clip maximum. That uses the video, not the EMG, so the window is not biased toward the EMG you are measuring.
2. Per channel, take the median envelope over the hold minus the `Face-At-Rest` median for that session.
3. Normalize each channel by a **reference contraction**, the 95th percentile of that channel across all `schaede` clips in the session (a stand-in for MVC, which MMM does not have). Report the max-normalized version as a sensitivity check.
4. Average left and right into 11 muscle sites for the main analysis. Keep both sides for AU12L and for the asymmetry check.

Output: a table of task x site x session, plus a heatmap.

Reliability: Spearman ρ of each task's vector between `1S` and `2S`, and ICC per site across tasks.
This sets the ceiling. The model cannot be expected to agree with the human more than session 1 agrees with session 2.

### Step 3. Model recruitment vectors (2 to 3 days)

- Add a small export to the evaluator: for each `AU0 -> pose` transition, write per-structure muscle work, strain energy and per-muscle passive length change to JSON. Leave the solver unchanged.
- Map the result to the 11 sites with the table in section 4.
- Run it for the 7 paired poses.

### Step 4. Metrics (1 day)

Per pose:

- **Agonist agreement**: is the human site with the highest normalized EMG one of the structures the model drives? A yes/no per pose, reported as k/7.
- **Rank agreement**: Spearman ρ between the model and human vectors across the 10 sites without the masseter. With only 10 points, report a permutation p-value and a bootstrap CI over frames in the hold window. Do not read much into a single pose's ρ.
- **Passive prime-mover count**: the number of poses whose human top-1 site is passive in the model.
- **Driven-but-silent check**: the model drives orbicularis oris in AU12. Does the human `Orbicularis Oris` channel rise in `Smile-Closed`? This is where crosstalk from the zygomaticus matters most, so read it next to the Lapatki 2010 electrode guidance.

Across poses:

- Pool all pose x site pairs and compute ρ, model against human, with session-to-session ρ as the ceiling.
- Repeat with the passive-deformation vector in place of the work vector. If passive deformation tracks EMG but drive does not, the model moves the right tissue for the wrong reason, which is exactly the point to make.

### Step 5. Cross-check against published group data (1 day)

- Take the group sEMG means for the matching movements from the Jena papers (Schumann 2021 for the Schaede exercises, Guntinas-Lichius 2023 for emotions).
- Check whether this participant's top sites match the group's. If they do, the single-subject result is more likely to hold in general. If they do not, say so.

### Step 6. Write-up (1 day)

- One figure: a heatmap of human against model recruitment, poses as rows and sites as columns, with passive model muscles hatched.
- One table: the metrics per pose with session ceilings.
- One paragraph per prediction from `VALIDATION_PLAN.html`.

## 6. Predictions, written down before running

- **Fail** for AU12 and AU6+12+25: the human top site will be `Zygomaticus`, which the model leaves passive.
- **Fail** for AU4 if it is used: corrugator and depressor supercilii/procerus are passive or absent.
- **Pass** for AU1+AU2 (frontalis), AU7 (orbicularis oculi) and AU22 (orbicularis oris), assuming step 1 resolves the labels.
- AU0 is a baseline, not a test.

Expected headline: about 3/6 poses with agonist agreement, all failures being prime movers the model leaves passive.
This argues for making activation explicit (option C in `VALIDATION_PLAN.html`).

## 7. Limits to state in the paper

- One participant. Session agreement measures reliability, not generalization.
- The 30 Hz, about 500 ms RMS envelope rules out timing claims.
- There is no MVC recording, so normalization relies on reference contractions within the protocol.
- Bipolar sEMG picks up neighbouring muscles, especially around the mouth: zygomaticus, orbicularis oris and levator labii.
- The model's "recruitment" is solver work under a constant activation of 0.8. It is a proxy, not a neural drive.
- Poses and tasks pair imperfectly: pucker is not AU22, and there is no single-AU task for AU1, AU2 or AU25.

## 8. Effort

About 9 to 10 working days in total, which fits the "about 2 weeks" in `VALIDATION_PLAN.html`.
Step 1 blocks everything else. Steps 2 and 3 can run in parallel.
