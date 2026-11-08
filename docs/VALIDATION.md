# Local validation, November 7, 2026

Implementation branch: `codex/implement-shield`, based on upstream commit `6eb224e27f4964b173cbe26deeb26620fc3aad95`. Changes are local. The connected GitHub account has pull access and no push access to the upstream repository.

Environment: macOS ARM64, Python 3.13.7, PyTorch 2.15.0, CPU, two Torch threads. See [machine-readable evidence](validation.json) for package versions and measured local benchmark results.

## Results

- `python -m pytest -q`: **20 passed** (final run: 1.90 seconds).
- `ruff check shield scripts tests`: all checks passed.
- Package installed successfully in editable mode; CLI training, federated training, evaluation, standalone inference, ledger verification and checkpoint-anchor verification ran successfully.
- Full-width model (base channels 64, 256-D ISV, 220 classes) passed a forward/shape check. It has 14,568,413 parameters with that classifier size. The small demo has 244,101 parameters.
- Two centralized epochs and two Dirichlet-partitioned FedAvg rounds completed on 16 training images. Three validation images are from a disjoint patient. Retrieval has three additional patients, six enrollment images and six held-out query images.
- Four noise levels produced 24 reconstructed query images; every saved shard manifest was checked and reassembled exactly. A subsequent mixed-corruption run exercised sigma=.05 plus 15% masking on six queries.
- The final central provenance ledger contains 68 verified blocks: 2 model anchors, 6 enrollment events across the two evaluations, 30 reconstruction events and 30 verification events. The FL ledger contains two global-model anchors.
- Real MPI ran with four separate processes in clean, missing-vote and conflicting-signature scenarios. All three scenarios committed five blocks on every process; each scenario's four ledger heads matched. Fault runs withheld/corrupted one of four votes. They did not terminate a process or exercise view-change recovery.
- The serial local benchmark completed 10 transactions each at 20, 50 and 100 emulated validators with 10% conflicting-signature injection. These are explicitly serial-signing measurements, not the paper's physical-cluster scaling results.

## Synthetic evaluation outputs

These small-sample results check execution only. They are not medical dataset results and do not establish denoising or identity-verification effectiveness. The tiny model is close to its residual identity initialization after two epochs; no claim of improvement over the noisy input is made.

| Gaussian sigma | PSNR (dB) | SSIM | Rank-1 | Pairwise AUC |
|---|---:|---:|---:|---:|
| 0 | 46.55 | 0.9976 | 0.8333 | 0.9306 |
| 0.02 | 33.87 | 0.8975 | 1.0000 | 1.0000 |
| 0.05 | 26.61 | 0.6713 | 0.6667 | 0.8472 |
| 0.10 | 21.23 | 0.4313 | 0.6667 | 0.8333 |

Mixed sigma=.05 plus 15% masking: 16.79 dB PSNR, 0.5023 SSIM, Rank-1 0.5, AUC 0.6944. Only six query images were evaluated in each condition; variability and non-monotonic Rank-1 scores are expected at this sample size.

A portable copy of metrics and previews is under `examples/synthetic-demo/`. Complete local checkpoints, templates, shards, ledger records, and simulated private keys remain under `runs/final-demo/`; they are ignored by Git and excluded from the source archive. The archive includes no private keys, patient data, or model checkpoints. Preview images show clean / degraded / reconstructed synthetic inputs left to right.

## Commands used

An existing environment in this workspace can be activated from the repository root:

```bash
source ../../work/.venv/bin/activate
python -m pytest -q
python -m shield demo --output runs/another-demo
```

The completed run is `runs/final-demo`. Its current central manifest fingerprint and head are retained here:

```text
manifest: 8beb0cf954352b38c151bf6a732e7725b05d612d8ad01e935eda9692f64881c1
head:     e807e8e76de441b14da32e2478cdce058cd5c8a095055c54f69f7ce904ef8f7c
model:    24fa816fadc4471b30ea4ab560f61f5e4f950b13c9c5958912a500ecceb4d9af
```

```bash
python -m shield verify-ledger --ledger runs/final-demo/centralized/ledger \
  --manifest-hash 8beb0cf954352b38c151bf6a732e7725b05d612d8ad01e935eda9692f64881c1 \
  --head e807e8e76de441b14da32e2478cdce058cd5c8a095055c54f69f7ce904ef8f7c
```

A later evaluation appends blocks, so its new head supersedes this retained head. Keep trusted fingerprints outside the ledger when testing replacement/truncation detection.

## Still required for paper-level reproduction

Actual NIH ChestX-ray14 and BraTS 2020 cohorts, exact data selection and preprocessing decisions, full training across five seeds, the missing architecture/loss specifications, external baseline implementations, and the paper's cluster/network setup are required. No 27.9 dB medical reconstruction, 98.7% patient retrieval, 500-node latency, BFT safety, 1.04x ledger replication, or 0.6 KB provenance result is claimed by this validation.
