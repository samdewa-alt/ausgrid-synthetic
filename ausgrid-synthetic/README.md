# Ausgrid paired daily synthetic profiles

This reproducible project generates paired **total recorded consumption**
(`GC + CL` when the household has a CL record) and **gross solar generation**
(`GG`) as 48 half-hour kWh readings per day. The primary question is whether a
conservative solar-night constraint improves plausibility without sacrificing
fidelity, utility, or measured household disclosure risk.

## Colab quick start

1. Unzip the project archive into Google Drive so it becomes
   `MyDrive/ausgrid-synthetic/` with `src/` and `notebooks/` inside.
2. In `MyDrive/ausgrid-synthetic/data/raw/`, place the **three unchanged CSVs**:
   `Solar home 2010-2011.csv`, `Solar home 2011-2012.csv`, and
   `Solar home 2012-2013.csv`. Keep the original first descriptive row.
3. Open `notebooks/Ausgrid_Colab.ipynb` in Google Colab, choose a GPU runtime,
   and run the cells in order. Mount Drive when prompted. If Colab disconnects,
   rerun the notebook: it skips completed stages and resumes training from
   an epoch checkpoint. Set `PROJECT_ROOT` in its first code cell if needed.

No raw CSVs are included in this archive. Prepared data, trained model states,
synthetic samples, and JSON reports are written under this project folder in
Drive. A full run can use several GB; check available Drive space. Generate
the short two-epoch pilot first, then decide whether to proceed with all six
neural runs. No external API key or experiment tracker is required.

## Folder layout

```
ausgrid-synthetic/
  data/raw/                # original CSVs supplied by you
  data/prepared/           # paired arrays, household split, audit
  outputs/checkpoints/     # best and latest epoch checkpoints
  outputs/samples/         # generated day profiles by arm/context/seed
  outputs/reports/         # locked test reports
  notebooks/Ausgrid_Colab.ipynb
  src/ausgrid_synth/       # preprocessing, models, evaluation, CLI
  requirements.txt
```

## Data and evaluation protocol

- A household with **no CL records anywhere** contributes GC as its recorded
  consumption. A household that **ever has CL records** contributes `GC + CL`
  only on days with GC, GG and CL all present. Missing CL on such households
  never becomes zero. This fixed rule avoids changing the target definition
  when a CL row vanishes. Verify the implied selection in `audit.json`.
- Preserve the original interval order. `0:30` is the 00:00–00:30 interval;
  final `0:00` is 23:30–24:00 on the labelled date. Exclude the two local
  clock-change dates per year because the raw files contain 48 labelled slots
  even on daylight-saving transitions. Do not interpolate those dates.
- Exclude any paired day where GC, GG, or (if applicable) CL has an estimated
  `Row Quality` flag, a missing value, or a negative interval. The first file
  lacks the quality column. Confirm actual exclusions in the audit.
- The split is 180/60/60 **households** for train/validation/test, stratified
  approximately by solar capacity using five groups. Every day of each
  household remains in only one split. All scaling and regression fits use
  train households only.
- Inputs are cyclic day-of-year, weekend indicator, and log recorded panel
  capacity, standardized using the training split. Household ID, postcode,
  weather, and test consumption are not generation inputs.
- The default mask marks interval midpoints whose approximate solar elevation
  at Sydney is below −9°. This is a deliberately conservative *night* rule,
  not a precise roof-level PV model. The supplied data contain small positive
  nighttime GG values (at most about 0.013 kWh in the initial audit), so
  report both measured nighttime energy and the count above **0.02 kWh**.
  Never impose a strict kWh cap directly from the kWp rating.
- `statistical`: joint PCA–Ridge residual Gaussian baseline, season/calendar
  and capacity conditioned; `vae`: conditional VAE; `vae_post`: the same VAE
  samples with night masking applied after generation; `vae_daylight`: the same
  VAE architecture trained with the night mask. For each neural arm use three
  seeds with a shared train/validation/test split. The postprocessed arm has
  **no new training run**.
- Use matched held-out contexts for fidelity; use the same 30,000 train
  contexts for real-trained versus synthetic-trained morning-to-afternoon
  Ridge prediction, tested on real held-out households. A bounded nearest
  synthetic-profile attack reports household membership AUC. It is one
  empirical risk probe, **not proof of privacy**. Only independently sampled
  days are generated; no continuity across weeks or persistent synthetic
  household identities are claimed.

## CLI (optional outside the notebook)

Run from the project root with `PYTHONPATH=src`. In order:

```bash
python -m ausgrid_synth.cli prepare
python -m ausgrid_synth.cli baseline
python -m ausgrid_synth.cli train --arm vae --seed 1 --epochs 30
python -m ausgrid_synth.cli train --arm vae_daylight --seed 1 --epochs 30
for context in train test privacy; do
  python -m ausgrid_synth.cli sample --arm vae --seed 1 --context "$context"
  python -m ausgrid_synth.cli sample --arm vae_post --seed 1 --context "$context"
  python -m ausgrid_synth.cli sample --arm vae_daylight --seed 1 --context "$context"
done
python -m ausgrid_synth.cli evaluate --arm vae_daylight --seed 1
```

Repeat seeds 2–3 through the notebook. The statistical baseline uses seed 1.
`*_latest.pt` captures the optimizer and random generator state at each epoch;
`*_seedN.pt` retains the best validation model. Use validation for adjustments;
inspect the final test only after decisions are frozen. A future manuscript
should report the attack's limited power, the single geographical and
historical dataset, and selection of customers based on first-year data quality.
