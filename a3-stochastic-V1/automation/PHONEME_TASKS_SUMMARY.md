# Phoneme Transition Tasks Summary

## Sentences Analyzed

### Sentence 1: "You've not eaten it yet"
**Ignoring 'e' in "yet"**

Phoneme sequence: `Y → UW → V → N → AA → T → IY → T → N → IH → T → Y → T`

### Sentence 2: "new tea, new yacht, new TV"

Phoneme sequence: `N → UW → T → IY → N → UW → Y → AA → T → N → UW → T → IY → V → IY`

## Task Generation Results

### Tasks Already in Database (Skipped)
These tasks were already completed and stored in `deformation_path_db.json`:
- dataV22_AU0_neutral to dataV22_phoneme_AA
- dataV22_AU0_neutral to dataV22_phoneme_IY
- dataV22_AU0_neutral to dataV22_phoneme_N
- dataV22_AU0_neutral to dataV22_phoneme_T
- dataV22_AU0_neutral to dataV22_phoneme_UW
- dataV22_AU0_neutral to dataV22_phoneme_V

### New Tasks Generated (19 transitions - IH treated as IY)

1. **dataV22_AU0_neutral to dataV22_phoneme_Y** *(requested)*
2. dataV22_phoneme_Y to dataV22_phoneme_UW
3. dataV22_phoneme_UW to dataV22_phoneme_V
4. dataV22_phoneme_V to dataV22_phoneme_N
5. dataV22_phoneme_N to dataV22_phoneme_AA
6. dataV22_phoneme_AA to dataV22_phoneme_T
7. dataV22_phoneme_T to dataV22_phoneme_IY
8. dataV22_phoneme_IY to dataV22_phoneme_T
9. dataV22_phoneme_T to dataV22_phoneme_N
10. dataV22_phoneme_N to dataV22_phoneme_IY *(IH → IY substitution)*
11. ~~dataV22_phoneme_IH to dataV22_phoneme_T~~ *(removed - duplicate of IY to T)*
12. dataV22_phoneme_T to dataV22_phoneme_Y
13. dataV22_phoneme_Y to dataV22_phoneme_T
14. dataV22_phoneme_N to dataV22_phoneme_UW
15. dataV22_phoneme_UW to dataV22_phoneme_T
16. dataV22_phoneme_IY to dataV22_phoneme_N
17. dataV22_phoneme_UW to dataV22_phoneme_Y
18. dataV22_phoneme_Y to dataV22_phoneme_AA
19. dataV22_phoneme_IY to dataV22_phoneme_V
20. dataV22_phoneme_V to dataV22_phoneme_IY

## Statistics

- **Total phoneme transitions (with duplicates):** 28
- **Unique transitions after deduplication:** 19 (IH treated as IY)
- **Tasks skipped (already in DB):** 0 new transitions matched existing DB entries
- **Tasks added to pipeline:** 19

## Deduplication Details

The following transitions appeared multiple times in the phoneme sequences but were only added once:
- **T → IY**: Appears 3 times in combined sequences
- **UW → T**: Appears 3 times
- **N → UW**: Appears 3 times
- **Y → UW**: Appears in both sentence starts
- **V → N**: Common transition
- And others...

## Implementation

All 19 tasks have been added to `automation/run_pipeline_update.py` at lines 172-193.

The pipeline will process these tasks sequentially, with each task depending on prerequisite tasks being completed first (for non-neutral starting poses).

## Phoneme Coverage

### Phoneme Substitution
- **IH** (as in "it") is treated as **IY** - no separate IH phoneme data available

### Phonemes Already Covered
- Y, UW, V, N, AA, T, IY, V (all already have neutral-to-phoneme transitions in DB)

## Expected Processing Order

The pipeline will intelligently handle dependencies:
1. First task starts from neutral → Y (no prerequisites)
2. Subsequent tasks starting from non-neutral poses will:
   - Check if the "from_pose" has a neutral-to-pose transition in the DB
   - Use that result to generate the rest pose
   - Then run the actual transition
