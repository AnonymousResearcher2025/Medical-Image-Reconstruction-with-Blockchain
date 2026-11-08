# SHIELD: Medical Image Reconstruction with Signed Provenance

Runnable research implementation of the supplied SHIELD manuscript: semantic reconstruction, patient identity vectors, Hilbert fragmentation, a trust-weighted scale-free overlay, Ed25519 provenance certificates, and federated checkpoint anchoring.

**Status:** implemented and exercised on synthetic data. This is an independent implementation from the manuscript, not the authors' released training code or a reproduction of its reported clinical/cluster results. Important ambiguities and counterexamples to two claimed properties are recorded in [paper notes](docs/PAPER_NOTES.md).

## Install and run

Use Python 3.10+; Python 3.12 is a convenient default. Run commands from this repository's root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,medical]'
python -m pytest -q
python -m shield demo --output runs/my-demo
```

The demo needs no dataset download or GPU. It generates synthetic patterns and runs two centralized epochs, enrollment/query evaluation at four noise levels, and two FedAvg rounds across two simulated clients. `configs/demo.yaml` uses an 8-channel miniature model; `configs/paper.yaml` uses the full 64-channel ResNet-18-width encoder and all 64 shards of each 256x256 image. The demo is intended to finish quickly and is not an accuracy benchmark. Each output directory must be fresh; completed runs are not overwritten or silently resumed.

On the machine where this implementation was built, an installed environment is also available at `../../work/.venv`. Activate it with `source ../../work/.venv/bin/activate` instead of reinstalling. Verified local outputs and the machine-specific validation report are under `runs/final-demo` and [docs/VALIDATION.md](docs/VALIDATION.md).

## Data manifest and leakage controls

Supply a CSV with `path,patient_id,split`. Image paths are relative to the CSV or absolute. Accepted splits are `train`, `val`, `enroll`, and `query`.

```csv
path,patient_id,split
images/train001.png,train-p001,train
images/train002.png,train-p001,train
images/validation001.png,val-p001,val
images/enrollment001.png,retrieval-p001,enroll
images/query001.png,retrieval-p001,query
```

Training and validation patients must be disjoint. Retrieval patients must be disjoint from both. Enrollment and query may share a patient but never an image, and every query patient needs enrollment. Multiple patients with multiple acquisitions each are needed for meaningful identity/triplet training. Use pseudonymous identifiers. Data are read locally; datasets are never automatically uploaded or downloaded.

**NIH ChestX-ray14:** obtain the authorized dataset separately, and point to its metadata CSV and a flat images directory. The helper selects seeded, patient-disjoint cohorts and reserves 15% of reconstruction patients for validation. Cohort counts are explicit; it does not claim the paper's unpublished exact image selection.

```bash
python -m shield prepare-nih --metadata data/nih/Data_Entry_2017.csv \
  --images data/nih/images --output data/nih/manifest.csv \
  --train-patients 500 --retrieval-patients 220 --seed 0
```

**BraTS 2020:** install the `medical` extra. The helper reads local `*_t1ce.nii.gz` files, selects cases, exports central axial slices 60..79, and splits by case. These are 2-D intensity denoising experiments, not MRI k-space reconstruction.

```bash
python -m shield prepare-brats --root data/BraTS2020 \
  --output data/brats-slices --cases 50 --seed 0
```

## Train, evaluate, and reconstruct

Review `configs/paper.yaml` and adjust memory use, hardware and cohorts. `device: auto` chooses CUDA when available and otherwise CPU. `device: mps` can be selected explicitly on compatible Macs. `tiles_per_image: 0` uses every shard; a positive value subsamples training shards to reduce memory. The full configuration can be expensive: eight images times 64 shards is 512 tiles per training batch.

```bash
python -m shield train --manifest data/nih/manifest.csv \
  --config configs/paper.yaml --output runs/nih-central
python -m shield evaluate --manifest data/nih/manifest.csv --run runs/nih-central
python -m shield evaluate --manifest data/nih/manifest.csv --run runs/nih-central \
  --sigmas 0.05 --mask-fraction 0.15
python -m shield infer --checkpoint runs/nih-central/model.pt \
  --image data/nih/images/example.png --output runs/example-reconstruction.png
```

Evaluation enrolls clean reconstructed images, averages their shard ISVs per patient, verifies template hashes and model bindings, and ranks each reconstructed held-out query by cosine similarity. Each reconstruction and verification query gets a ledger event. Raw reconstructed shards go into per-owner directories with content-addressed manifests and Merkle commitments. Evaluation reports PSNR, local-window SSIM, ISV cosine, Rank-1 and ROC AUC; the preview contains clean / degraded / reconstructed images from left to right. Re-running evaluation adds events and updates its report. Mixed-mask evaluations use a separate `evaluation-mask-0.15.json` report and matching preview.

Standalone `infer` checks the checkpoint's internal tensor hash and writes an image; it reports `anchored: false`. Use `evaluate` for the full anchored pipeline, or separately verify the checkpoint's anchor before standalone inference.

Loss terms follow Eq. 2. Weights and optimizer rates are configurable. The implementation explicitly selects details missing from the manuscript, including semantic lambda, loss distances, GroupNorm and triplet mining; see [paper notes](docs/PAPER_NOTES.md).

## Federated training

```bash
python -m shield federated --manifest data/nih/manifest.csv \
  --config configs/paper.yaml --output runs/nih-federated
```

Set `federated.clients` to 5 or 10, `rounds: 50`, `local_epochs: 5`, and `partition: iid` or `dirichlet` with `alpha: 0.5`. Partitions are deterministic, disjoint, and saved. Clients run sequentially in the local simulator. Uniform client weighting implements Algorithm 2; `sample_weighted: true` is an optional sample-weighted FedAvg extension. A global checkpoint and its client-update hashes are anchored after every round. This is not a networked hospital training service.

## Provenance verification

`training.json` and `evaluation.json` include a manifest fingerprint and head hash. Retain these outside the ledger directory to make replacement/truncation detectable. The manifest pins validator public keys, topology and the fault parameter.

```bash
python -m shield verify-ledger --ledger runs/nih-central/ledger \
  --manifest-hash RETAINED_MANIFEST_SHA256 --head RETAINED_HEAD_SHA256
python -m shield verify-checkpoint --checkpoint runs/nih-central/model.pt \
  --ledger runs/nih-central/ledger --manifest-hash RETAINED_MANIFEST_SHA256
```

Hashes alone do not authenticate a substituted manifest. A valid certificate under the paper's Eq. 4 also does not establish BFT agreement: the rule can admit disjoint quorums. The local ledger is a single-writer simulation with co-located test keys. Do not deploy it as a clinical security boundary. Tests cover forged signatures, duplicate signers, insufficient weight, vote conflicts, altered templates, altered shards, cohort leakage, aggregation and model reloads.

## Real MPI signature collection

Install `mpi4py` with an MPI runtime. For a local development runtime, the mpi4py project documents `python -m pip install mpi4py mpich`; clusters can use their existing OpenMPI/MPICH module.

```bash
python -m pip install mpi4py mpich
mpiexec -n 4 python scripts/mpi_consensus.py --output runs/mpi --transactions 10
mpiexec -n 4 python scripts/mpi_consensus.py --output runs/mpi-conflicts \
  --transactions 10 --fault byzantine --fault-rate 0.25
```

Each process has its own ephemeral Ed25519 key. Votes travel through real MPI Allgather and committed blocks through Bcast; every rank validates its own ledger copy. `--fault crash` withholds votes; `--fault byzantine` signs a conflicting payload. These inject application faults, not killed MPI processes. Actual process failure can stall collectives; view changes and crash recovery are not implemented. This MPI path tests consensus certificates, not distributed GPU inference/training. Audit proofs are fully replicated, so it does not claim the paper's sharded-ledger storage factor.

A separate local benchmark records real serial signing/commit timings, clearly labeled as such:

```bash
python -m shield benchmark-consensus --output runs/local-bench \
  --nodes 20 50 100 --transactions 10 --fault byzantine --fault-rate 0.1
```

## Repository layout

| Path | Purpose |
|---|---|
| `shield/model.py`, `losses.py` | ResNet-stage U-Net, ISVs, composite reconstruction/identity loss |
| `shield/gft.py`, `shards.py` | Deterministic placement and verified off-ledger shard storage |
| `shield/overlay.py`, `ledger.py`, `crypto.py` | Scale-free topology, signatures, quorum certificates, hash chain |
| `shield/store.py` | Content-addressed template enrollment and authenticated cosine retrieval |
| `shield/data.py` | Patient manifests, corruptions, NIH/BraTS preparation, synthetic data |
| `shield/engine.py`, `federated.py` | Training, metrics, checkpoint anchoring, FedAvg |
| `scripts/mpi_consensus.py` | MPI certificate pipeline |
| `tests/` | Integrity, numerical, data and end-to-end regression tests |
| `docs/PAPER_NOTES.md` | Manuscript mapping, assumptions, counterexamples and reproduction limits |

The repository retains its upstream MIT license. No pretrained model or medical dataset is distributed. SwinUNet/TransUNet/DPS and PBFT/HotStuff baseline implementations are outside this release; their published results are not presented as generated outputs.
