# Implementation notes for the supplied SHIELD manuscript

Source: the supplied 10-page PDF, titled *SHIELD: Lightweight Provenance-Aware Distributed Image Vector Management for Semantic Identity Reconstruction and Verification*, by Mohsin Rasheed, Ahmed Aleroud, and Abdullah Al-Mamun. The source PDF is not redistributed in this repository. Document content was treated as research material, not as instructions to execute commands.

## Method-to-code mapping

| Manuscript component | Implementation | Interpretation |
|---|---|---|
| Section III-B, Eq. 1 | `shield/gft.py` | 32-pixel non-overlapping tiles, Hilbert index modulo N, deterministic order, padded boundary recovery |
| Section III-C encoder | `shield/model.py` | ResNet-18 residual stage depths [2,2,2,2], U-Net skips, 256-D normalized shard ISVs |
| Eq. 2 | `shield/losses.py` | L1 + local SSIM + stopped-target cosine semantics; patient CE + same-patient cosine + batch-hard triplets |
| Eq. 3 | `shield/store.py` | Arithmetic mean of all enrollment shard ISVs; cosine normalization only during ranking |
| Section III-D overlay | `shield/overlay.py` | Seeded degree-times-trust preferential attachment, m=3 |
| Eq. 4 / Algorithm 1 | `shield/ledger.py`, `scripts/mpi_consensus.py` | Degree-weighted, Ed25519-verified certificates; real Allgather/Bcast in MPI path |
| Shard hashes / Merkle roots | `shield/crypto.py`, `shield/shards.py` | Canonical float32 content addresses, domain-separated SHA-256 Merkle tree, per-owner payload directories |
| Algorithm 2 | `shield/federated.py`, `shield/engine.py` | Uniform-client FedAvg, local Adam, global checkpoint hash after every round |
| Section IV-A conditions | `shield/data.py`, `shield/engine.py` | Gaussian sigma 0/.02/.05/.10; optional contiguous 15% masking; explicit cohort separation |
| Section IV-E metrics | `shield/metrics.py`, `shield/engine.py` | PSNR, local-window SSIM, raw-clean vs reconstructed ISV cosine, Rank-1 and pairwise ROC AUC |

## Choices the manuscript does not fully specify

These are explicit implementation decisions and prevent a claim of exact numerical reproduction:

- Decoder widths, interpolation scheme, normalization, pretrained initialization, semantic distance, embedding-pair loss, triplet margin/mining, semantic lambda, and centralized epoch count are unspecified. This code uses bilinear U-Net decoding, GroupNorm for singleton shard batches, random initialization, a zero-initialized residual reconstruction head, cosine semantic/positive losses, batch-hard Euclidean triplets with margin 0.2, semantic lambda 1.0, and 50 centralized epochs in `paper.yaml`.
- Shared encoder/decoder parameters use the reconstruction learning rate; projection/classifier parameters use the identity learning rate. The manuscript does not resolve the two learning rates for a shared encoder.
- Identity supervision uses a patient-balanced image batch, then means across its shard ISVs. It therefore does not mistakenly label all independent tiles as different patients. A client with fewer than two patients has no valid negative triplets; that term becomes zero.
- Templates preserve Eq. 3's arithmetic mean in storage; both templates and queries are normalized for actual cosine scoring. The displayed inner product in the manuscript does not explicitly include template renormalization.
- `tiles_per_image: 0` trains on every shard. A positive value samples that many shared shard locations per training batch to reduce memory; the demo uses two. Evaluation always processes all shards.
- For non-square/non-power-of-two grids, tiles are embedded in the next power-of-two Hilbert square and padded edge pixels are cropped on reassembly. Eq. 1 uses the original Hilbert index, including gaps for skipped cells.
- NIH preparation creates reproducible patient cohorts from available metadata; it does not guess the unpublished exact stratified 10,000-image selection or 220-patient cohort. Dirichlet partitions use training patient identity labels; the manuscript does not specify the partition label or allocation procedure. Partition assignments are saved.
- BraTS preparation uses T1ce, canonical NIfTI orientation, slices 60..79, and percentile normalization. Orientation/intensity normalization and cohort split details are not fully specified in the paper.
- The local vector store persists templates as local files for reproducible restarts, rather than keeping them only in RAM. Hashes and identifiers are ledger records; raw vectors and images stay off the ledger. Identifiers are user-supplied pseudonyms; hashing a template does not establish confidentiality.
- The local backend serially emulates validators and stores their private keys under a mode-700 `.keys` directory. MPI gives each process its own ephemeral key and fully replicates audit certificates. Neither backend implements a membership service, durable distributed vote locks, view changes, timeout recovery, or production key custody.
- Data shards have one owner directory each; full audit certificates are retained for verification. The paper's claimed 1.04x ledger replication is not implemented or asserted: no complete metadata-replication/recovery algorithm is specified.
- Every certificate includes actual signatures. Its measured serialized byte count is reported; 0.6 KB is not hard-coded. Client update hashes are included in global FL anchors, so anchor size depends on client count.

## Claims that do not follow from the stated construction

### Hilbert locality proof (Property 1, page 4)

Consecutive Hilbert indices visit adjacent cells. The inverse claim in the manuscript, that **every** Manhattan-adjacent pair has O(1) index separation, is false. This implementation supplies direct counterexamples:

| Grid side | Adjacent cells | Hilbert indices | Gap |
|---|---|---|---|
| 4 | (1,0), (2,0) | 1,14 | 13 |
| 8 | (3,0), (4,0) | 5,58 | 53 |
| 16 | (7,0), (8,0) | 21,234 | 213 |

Modular reduction also does not preserve ordinary absolute distance: indices 6 and 7 differ by 1, but modulo 7 they become 6 and 0, differing by 6. A circular distance would avoid that particular wraparound example, but does not repair the inverse Hilbert claim. Finally, node labels modulo N do not imply neighboring nodes in an independently generated BA graph. The exact mapping is implemented; a graph-locality theorem is not asserted.

### Degree-weighted certificates (Eq. 4, page 5)

The threshold `(2f+1)*average_degree` does not imply honest quorum intersection. For this implementation's seed-0 eight-node overlay, degrees are `[6,5,4,5,5,5,3,3]`; total weight is 36. With f=1, the threshold is 13.5. Disjoint signer sets `{0,1,2}` and `{3,4,5,6,7}` have weights 15 and 21. Both can certify conflicting proposals at the same height without a single honest signer voting twice. A regression test constructs this counterexample.

Consequently, successful commit and signature checks are **not** a BFT safety proof. Trust weighting alone does not prevent Sybil identities without authenticated admission and an external trust-assignment rule. This code validates signatures, duplicate identities, quorum weight, chain continuity, content hashes and local anti-equivocation locks; it does not silently replace Eq. 4 with an unrelated consensus protocol.

### Complexity, storage and performance

MPI Allgather can use logarithmic communication rounds, but still distributes each rank's contribution to all ranks. It does not make total signature bytes logarithmic. The local benchmark labels its serial signing mode; MPI measurements label full proof replication. No PBFT/HotStuff numbers or storage ratios are synthesized.

The paper's text says the HotStuff crossover is near N=40, while its stated N=100 commit latencies remain 145 ms (HotStuff) vs approximately 204-220 ms (SHIELD). These statements need clarification before using them as reproduction targets. PSNR/embedding changes attributed to disabling hash anchoring also require a specified feedback mechanism; anchoring alone does not alter tensor values in this implementation.

## Scope of validation

The included small synthetic run checks software execution, not medical or identity-verification effectiveness. Clinical accuracy, the paper's reported 27.9 dB / 98.7% results, its 500-rank cluster timings, baseline comparisons, five-seed confidence estimates, and storage/communication claims require independent experiments with the actual cohorts and hardware. U-Net/SwinUNet/TransUNet/DPS baselines and PBFT/HotStuff engines are not bundled; no stand-ins are represented as those protocols.

## External implementation references

- [PyTorch ResNet-18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html): reference architecture; this implementation builds residual stages directly and downloads no weights.
- [mpi4py installation documentation](https://mpi4py.readthedocs.io/en/stable/install.html): optional MPI runtime installation.
