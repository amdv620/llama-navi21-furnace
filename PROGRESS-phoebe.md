# llama-navi21-furnace on `phoebe` — progress log

Fork of `ggml-org/llama.cpp` tuned for **2× AMD Radeon PRO V620** (Navi21 / gfx1030 /
RDNA2, 32 GB each) on a single-user R&D box.

| | |
|---|---|
| Cloned | 2026-09-19 20:13 UTC |
| Fork point | `e613ef2c8` — *hexagon: enable I32 GET_ROWS (#29116)*, upstream master 2026-09-19 |
| Branch | `tune/gfx1030-mmq` — 18 commits |
| Host | Gigabyte X570 AORUS PRO WIFI (BIOS F34), Ryzen 7 5800XT, Ubuntu 26.04, kernel 7.0.0-31, ROCm 7.1 |
| Status | Software work complete and committed. **Hardware currently down** — see *Open: boot*. |

Headline numbers, single V620, Qwen3.8-27B class model, `-fa on`:

| metric | at clone | now | change |
|---|---|---|---|
| pp512 (prefill) | 437.1 t/s | 442.3 t/s | +1.2% |
| tg128 (decode) | 23.15 t/s | 23.79 t/s | +2.8% |
| decode + speculation | — | ~50 t/s mean, 67 t/s on math | ~2.1× |
| flash-attn (`-fa on`) | unusable | all model shapes working | — |

---

## 1. Hardware bring-up

The cards were dead on arrival to ROCm: `amdgpu` spun forever on
`trn=2 ACK should not assert! wait again !`, KFD exposed no compute nodes, and each
V620 showed only its 2 MB doorbell and 512 KB MMIO BARs with **BAR0 — the 32 GB VRAM
aperture — entirely unassigned**.

**Fix: disable CSM in BIOS.** With CSM off the kernel places both 32 GB BARs, KFD comes
up with `simd_count 144`, and llama.cpp sees ROCm0/ROCm1 at 32752 MiB each, PCIe Gen4 x16.

Wrong turns worth not repeating:

- It is **not** a VBIOS or SR-IOV firmware problem. An earlier session concluded the cards
  needed reflashing; the decisive counter-evidence was that the same cards ran fine in a
  different machine.
- `trn=2 ACK` is not SR-IOV-specific — it shows up on bare-metal Vega/MI25/Radeon VII too,
  and is a downstream symptom of the failed VRAM aperture, not a cause.
- Do **not** force PCIe Gen3. BAR placement is an address-space problem, not a link-speed
  one, and Gen3 would halve bus bandwidth for nothing.

Also set, belt-and-braces, before the CSM change and never individually verified as
necessary: `GRUB_CMDLINE_LINUX="pci=realloc=off amdgpu.gpu_recovery=1 amdgpu.mcbp=0"`.

`~/v620-check.sh` re-verifies the whole chain in one command.

### Measured hardware characteristics

`rocm-smi` reports mclk "1000 MHz" (memory-*controller* clock) against TechPowerUp's
"2000 MHz" (effective). Same physical GDDR6 at 16 Gbps — **not** a half-speed bug, do not
re-chase it. Confirmed by direct microbenchmark:

- pure read: **505.6 GB/s** (~99% of the 512 GB/s spec)
- streaming copy (read+write): 419 GB/s

---

## 2. Build

ROCm 7.1 lives under `/usr`, not `/opt/rocm`. No `g++`, only `clang++-21`. CMake 3.31
rejects the hipcc wrapper and Ubuntu multiarch hides the HIP cmake package, so:

```bash
export ROCM_PATH=/usr HIP_PATH=/usr
CC=/usr/lib/llvm-21/bin/clang CXX=/usr/lib/llvm-21/bin/clang++ \
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DCMAKE_HIP_COMPILER=/usr/lib/llvm-21/bin/clang++ \
  -DCMAKE_HIP_FLAGS="--rocm-path=/usr --rocm-device-lib-path=/usr/lib/llvm-21/lib/clang/21/amdgcn/bitcode" \
  -DCMAKE_HIP_COMPILER_ROCM_LIB=/usr/lib/x86_64-linux-gnu \
  -DCMAKE_HIP_LIBRARY_ARCHITECTURE=x86_64-linux-gnu \
  -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/cmake \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j16
```

To compile-check one kernel without a full build, call `hipcc` directly with
`--offload-arch=gfx1030`.

---

## 3. Flash attention — from unusable to complete

Five commits took the gfx1030 tile path from aborting on every real shape to fully working.
The bugs were RDNA occupancy falling to 1 in the config tables, plus missing dispatch caps.

| commit | fix |
|---|---|
| `a63536900` | RDNA2 tile config fix (D=256/512) |
| `3ceecec3a` | extend to D=128 |
| `9b1ceb1fc` | D=576 occupancy abort (DeepSeek MLA) |
| `a02ddd408` | D=256 MHA, `ncols2=1` occupancy abort |

Validated with `test-backend-ops -o FLASH_ATTN_EXT`: D=128/256/512/576, KV type
f16/q8_0/q4_0, all batch sizes, KV length a multiple of 256.

**The real inference path fully works.** That 256 constraint is not a limitation in
practice — `llama_kv_cache::get_n_kv()` always pads `n_kv` to a multiple of 256
(`llama-kv-cache.cpp:1255`), so it is what the models actually produce.

Measured value of the tile fix: tg64 @8k 19.87 → 22.33, @16k 17.20 → 21.12,
pp512 @8k 268 → 363.

### Known remaining gaps (deliberately not fixed)

True out-of-bounds memory faults remain on: unaligned `n_kv` (e.g. 113 — never occurs in
real inference), exotic KV dtypes (bf16/iq4_nl/q2_0), ALiBi (`max_bias>0`), and attention
sinks. Several are byte-for-byte upstream bugs, visible at D=40/64/72 which our changes
never touch. These are configurations our models do not generate; fixing them would be
hardening upstream's RDNA2 tile kernel broadly, not fork work.

---

## 4. Kernel work

### Landed

**`0aa20a02b` — MMVQ config tables had no RDNA2 entry.** `calc_nwarps` fell through to
`return 1`, so *every decode dispatch was a single wave32 workgroup*. Added an RDNA2 branch
(nwarps 4 for `ncols_dst<=4`, 2 for `<=8`) plus an RDNA2 case in `calc_rows_per_block`.
**tg128 23.15 → 23.58 (+1.9%).** Do not use nwarps=8 (worse) or nwarps=2 at
`ncols_dst==1` (aborts inside `ggml_cuda_mul_mat_vec_q` — a latent upstream bug worth filing).

**`3bccd8323` — delta-net concat took a scalar path.** A `ggml_transpose` before
`ggml_concat(dim=0)` failed `ggml_is_contiguous_to_3`, so a scalar kernel walked the time
axis with ~16× read amplification. `ggml_cont()` after the transpose routes it to ggml's
dedicated `cpy_scalar_transpose`, 4.4× better: `concat_non_cont` 46.69 ms → 0, replaced by
10.57 + 5.75 ms. **pp512 437.6 → 442.5 (+1.1%).**

**`d660b2491` — three latent correctness hazards** from the kernel audit. 0% perf, all gates
green: a NaN guard for `amax==0` in `quantize_mmq_q8_1`; a `static_assert` at
`fattn-tile.cuh:513/598` that would otherwise silently drop Q columns; and mixing `n_nodes`
into the CUDA-graph key (it previously returned `nodes[0]` alone, collapsing the LRU to one
entry).

**`b5f65d215`, `05512bb1d` — MMVQ tuning for the speculative verify shape.** nwarps 4→2 at
`ncols_dst` 2..4 (pp4 78.4 → 83.4); nwarps 2→1 and `rows_per_block` 2→4 at 5..8
(pp8 112.1 → 120.5, +7.6%; pp6 101.1 → 107.7).

**`1184c3f4a`, `e628efa74`, `30061aa39` — q8_1 activation quantize cache.** Roughly 497
quantize calls per graph eval over only 257 distinct `src1` — ~48% redundant. The cache
lives on `ggml_backend_cuda_context`, MMVQ path only, kill switch
`GGML_CUDA_NO_Q8_1_CACHE=1`. **tg128 23.47 → 23.79.** Two follow-up fixes were needed and
are described under *Bugs found the hard way*.

### Tried, measured, rejected — do not redo

This is the more valuable half of the record.

| attempt | outcome |
|---|---|
| **`K_vram` / `MMQ_ITER_K` as a tuning knob** | **Dangerous.** Raising it looks like +29% pp512 but it *silently skips half the K data* and the model emits garbage. `ITER_K` must equal the 256-element tile capacity; the loop strides by `ITER_K/qk` while the tile covers `MMQ_TILE_NE_K`. It is 256 in every arch config because it is structural, not tuned. Test suite missed it — most shapes have k≤256. **Always end-to-end check a suspicious MMQ speedup with real output.** |
| **MMQ stream-k on the Q8_0 fallback** (`54451989c`, reverted `6022d3e4e`) | Made the M=128 `in_proj_ba` GEMM **3× faster** and gave +0.5% pp512 — but **regressed MTP decode 36.9 → 34.7 t/s (−6%)**, reproducible ×4 at 42 °C. Suspected: stream-k's fixup pool perturbs CUDA-graph reuse. **Lesson: always re-measure the production decode path after an MMQ change; prefill-only benchmarking hid a 6% regression.** |
| **Wiring the MUL_MAT_ID crossovers into the dense MMVQ path** | Looked neutral end-to-end but an A/B of the exact range regressed pp7 90.1 vs 102.4 (−12%). Those crossovers were measured for MoE routing and do not transfer to dense matmul. |
| **Forcing VGPR count down for occupancy** | `amdgpu_num_vgpr(168)` *is* honored (187→168) where `__launch_bounds__`/`waves_per_eu` are inert — but pp512 dropped 27% and occupancy was **unchanged** (23.5% vs 23.4%). Occupancy here is **LDS-bound, not VGPR-bound**. Closes the occupancy line; any future work must cut the ~45 KB dynamic LDS. |
| **LDS-staging the q8_1 activations** | The theory was 3.78× activation-vs-weight traffic amplification. Implemented correctly (142 `ds_read` / 30 `ds_write` emitted, tests passed) and it was **~2× slower on every shape**. The amplification theory was wrong. |
| **`rows_per_block=8`** | Predicted to fail on register pressure. It actually uses **fewer** VGPRs (168 vs 200) and gets **higher** occupancy (25.1% vs 22.0%) — and is still slower. Both of my stated mechanisms were wrong. |
| **Register-caching `x` in rms_norm** | Zero change. The kernel launches one 1024-thread workgroup at decode — 2.8% of the GPU, ~12 GB/s — so it is latency-bound, not traffic-bound. Halving reads cannot help. |
| **`{RMS_NORM,SCALE}` fusion** | Declined. Priced at 0.28–0.42% of decode. Also: `ggml_l2_norm` looks like a drop-in for `build_gdn_l2_norm` but the CUDA kernel computes `rsqrt(fmax(sum, eps*eps))` vs the required `rsqrt(sum + eps)` — **not equivalent**, would silently change numerics. |
| **CUDA-graph fork-point fix** | Audit claimed 112 fork points with 2 invalid discarding all. Instrumented reality: only 16 are built and **all 16** fail `is_valid()` on overlapping `Kcur` view writes. The fold is a real latent bug but fixing it yields 0 streams and 0% here. |

**Where the remaining headroom actually is.** `mul_mat_vec_q` runs at **480 of 505 GB/s =
95% of read roofline during its own execution**. An earlier "406 GB/s = 80%" figure divided
the same bytes by the whole token wall-clock — the decode gap is *not* in the GEMV inner
loop. Prefill is 88.3% `mul_mat_q` and 0.6% dispatch overhead. One decode dispatch costs
2.13 µs in situ, so removing N dispatches/token is worth N × 0.0049% of tg.

The only untried lever is widening the weight loads in `vecdotq.cuh` — the Q4_K
`ncols_dst=8` kernel issues **220 `global_load_dword` + 120 `global_load_ushort` and not a
single `dwordx2`/`dwordx4`**; the memory unit saturates on request count long before bytes.
That file is shared by every backend and quant type, so it is high blast radius and needs
explicit buy-in.

---

## 5. Speculative decoding

Two drafters were evaluated end-to-end.

**MTP** (`--spec-type draft-mtp`): single chained head. Per-iteration anatomy at n_max=3
(~60 ms, 2.40 tokens): target verify decode 52 ms = 85%; drafter 8.2 ms = 14%; acceptance
1.40 per iteration (46.5% of drafted). Both the drafter and the target's MMVQ sit at **98%
of the bandwidth roofline** — there is nothing left there.

The n_max sweep is done and the default is optimal: 1→36.4, 2→38.2, **3→38.5**, 4→37.0,
5→33.1, 6→29.9, 8→18.0. `p_min` at 0.4/0.6/0.8 all lose to n_max=3. Acceptance saturates at
~1.43 no matter how deep you draft.

**DFlash2** (`z-lab/Qwen3.8-27B-DFlash2`): block diffusion drafting, works out of the box in
this fork — no porting needed. Two setup notes: `n_max` defaults to 3 and **must** be raised
to `block_size-1 = 7` or the drafter is crippled to 3 of its 7 slots; and use the **Q4_0**
drafter, not Q8_0 (66.6 vs 63.3 t/s — Q4_0 is smallest and unpacks straight into dp4a).
`--spec-draft-p-min` is a **no-op** for dflash2 — the selector-confidence gate never fires.

Head-to-head, 256 tokens, greedy, Q4_0 drafters:

| prompt | MTP n=3 | DFlash n=3 | DFlash n=7 |
|---|---|---|---|
| GSM8K-ish math | 56.3 | 54.6 | **67.7** |
| code gen | 47.9 | **54.8** | 41.4 |
| boilerplate | 53.8 | 53.7 | **55.0** |
| reasoning essay | 40.0 | **40.4** | 31.9 |
| mean | 49.5 | 50.9 | 49.0 |

DFlash n=7 swings +20%/−20% with prompt entropy. An oracle picking n per prompt averages
54.5 — so adaptive depth was the single biggest remaining win, and since the shipped `p_min`
mechanism is dead it had to be built.

### Adaptive draft length (`d6bc22ebd`, `291449c0b`)

Per-sequence EMA of accepted draft tokens (α=0.2); next draft sized as
`clamp(lround(ema*1.3)+1, 1, cap)`. The cap is published on `dp.n_max`, the pre-existing
per-call override, so DFlash and MTP size the draft **up front** rather than truncating.
Kill switch `LLAMA_SPEC_NO_ADAPTIVE=1`. Gated to `cap >= 4`, so MTP at its optimal cap of 3
is bit-identical to before.

**DFlash cap 7: 53.1 adaptive vs 48.8 fixed n=7 (+8.9%)** over 4 workloads.

`291449c0b` fixes a real bug: the EMA carried over between server requests, so a math prompt
following an essay ran 71.8 vs 77.1 fresh. Now reset to the cap in `common_speculative_begin`.

Three variants were tried and lost — do not re-add: post-hoc truncation (49.8; the drafter
has already paid for discarded positions), censoring correction (52.9), and a hysteresis
deadband (51.5). The plain form won at 53.6.

---

## 6. Quantization

Built custom quants of `Jackrong/Qwopus3.8-27B-Flash-V2`, a fine-tune of Qwen3.8-27B.

Replicating an Unsloth UD recipe: `llama-quantize --tensor-type-file` takes `pattern=type`
lines, lowercased and applied with `std::regex_search`, so anchor and escape them. **Correct
ggml type ids: 20=IQ4_NL, 21=IQ3_S, 23=IQ4_XS** — 21/23 were swapped in the first attempt,
silently producing 70 tensors at 3.4 bpw where the reference uses 4.25. Always re-dump the
built file and diff the type histogram against the reference.

Per the no-IQ-in-the-target rule:

| build | size | pp512 | pp8 | tg128 |
|---|---|---|---|---|
| base Qwen3.8-27B UD-Q4_K_XL | 16.34 | 455 | 120 | 23.92 |
| Qwopus, UD recipe with IQ tensors → Q4_K | 16.57 | 431 | 108.5 | 23.81 |
| Qwopus, above with all Q4_K → Q5_K | 17.84 | 420 | 108.4 | 22.43 |

Dropping IQ costs ~10% on pp8 and nothing on tg128 — at `ncols_dst=8` IQ4_XS is 2.1× faster
than Q4_K, and 70 tensors × ~0.11 ms ≈ 7 ms of a 67 ms batch-8 eval, which matches the gap
almost exactly. A KL-divergence study confirmed the quality cost is negligible: mean KLD
0.00685 (no IQ) vs 0.00707 (with IQ), top-1 96.60% vs 96.51% — within error. The Q5_K
variant is the only real quality step (0.00512, 97.03%) and costs ~6% decode for +1.27 GiB.

**Key finding: DFlash2 does not transfer to a fine-tune; MTP does.**

| target | drafter | math | code |
|---|---|---|---|
| base | DFlash2 | 67.5 | 54.7 |
| base | MTP | 56.1 | 47.9 |
| Qwopus FT | DFlash2 | 54.8 | 43.6 |
| Qwopus FT | MTP | **55.9** | **46.2** |

DFlash2 loses ~19%/20% and its entire advantage. Hypothesis, untested: DFlash2 injects
hidden states from target layers [6,20,34,48,62], coupling it to internal representations
that a fine-tune shifts; MTP only consumes the final hidden state, a far more stable
interface. **So: DFlash2 for the stock model, MTP for fine-tunes.**

Extracting the fine-tune's own MTP head (`~/models/extract_mtp_head.py`) turned out to be a
no-op — this fine-tune never retrained the head. Its `blk.64` F32 norms are **bit-identical**
to the base drafter's, and the quantized ones differ by exactly Q4_K round-trip error. Worth
doing only for a fine-tune that actually trains its head; the correlation check costs seconds.

---

## 7. Bugs found the hard way

**Reference-logits memory fault.** `HSA_STATUS_ERROR_MEMORY_FAULT` during KL-divergence runs.
I first blamed host memory pressure and mmap, and was wrong. Bisection showed it needed the
q8_1 cache **and** CUDA graphs **and** multiple sequences together. Running with `-v` showed
**1988 "ROCm buffer pool full"** messages: the cache was unbounded, the legacy pool has 256
free slots, overflow triggered `cudaFree` on memory a captured graph still referenced →
page fault on replay. Fixed in `30061aa39` by bounding the cache to 8 entries and adding the
stream to the key. Validated: 0 faults, bit-identical perplexity 5.0177, tg128 23.72/23.67
with cache on vs 23.38/23.39 off.

**Cache outliving the pools.** The q8_1 cache tripped `GGML_ASSERT(pool_size == 0)` at exit.
Fixed in `e628efa74` by clearing it in the context destructor.

**Batch 8 → 9 cliff (open).** Decode ms by batch: 8 → 72.9, **9 → 108.2**. `MMVQ_MAX_BATCH_SIZE 8`
hands `ne11>=9` to MMQ, and MMQ is catastrophic on this path — forcing it at `ne11<=3` drops
MTP 38.7 → 25.1 t/s (−35%). Extrapolated MMVQ would not cross MMQ until batch ~13–14, so
batches 9–13 leave ~30 ms/eval on the table. Raising the limit means template instantiation,
not a runtime knob (`GGML_CUDA_MMVQ_MAX=64` aborts). Irrelevant at n_max=3; matters for
llama-server with parallel slots.

---

## 8. Current deployment

`~/models/serve-qwopus.sh` — Qwopus fine-tune (Unsloth UD-Q4_K_XL recipe with IQ tensors
moved to Q4_K, 16.57 GiB) + base MTP drafter at n_max 3, on GPU 1, bound to `0.0.0.0` for
LAN and tailscale, with MCP tools (`brave_web_search`, `brave_read`, `web_fetch`).

Measured: math 55.9 / code 46.1 / boilerplate 49.7 / prose 43.6 t/s.

---

## 9. Open items

**Boot (blocking).** The V620s are currently **out of the machine** — it will not POST
reliably with them installed. Root cause identified from the boot record: firmware is not
advertising an above-4G MMIO window, so the 32 GB BAR cannot be placed. When that happens
`amdgpu` probes anyway, reads all-ones from unmapped MMIO, concludes it is an SR-IOV virtual
function, and deadlocks in `xgpu_nv_mailbox_trans_msg` waiting for a hypervisor that does not
exist — an uninterruptible sleep during module load, which is the unbootable machine.

Confirmed by a clean A/B on 2026-09-19, same kernel and BIOS, four hours apart:

| | boot -4 (hung) | boot -3 (worked) |
|---|---|---|
| root bus above-4G window | none | `0x840000000-0xffffffffff` |
| V620 BAR 0 (32 GiB) | `can't assign; no space` | assigned at `0x8000000000` |
| result | hung in probe | `Detected VRAM RAM=32752M, BAR=32768M` |

Fix: **Above 4G Decoding → Enabled**, **CSM → Disabled**. Note that a hang counts as a failed
POST, and the board restores defaults after repeated failures — so the hang keeps undoing the
fix, which makes the setting look flaky. Optional hardening: disable SR-IOV if the board
exposes it, which drops the bridge window requirement from 416 GiB (32 GiB PF + 384 GiB of VF
BARs for 12 VFs) to 32 GiB. Recovery lever: `modprobe.blacklist=amdgpu` at the GRUB prompt.

BIOS is **F34 (July 2021)** against a **Ryzen 7 5800XT (July 2024)** — the firmware predates
the CPU by three years and Linux is already replacing its microcode at boot
(`0x0a201204 → 0x0a201211`). Latest is **F40c**. Worth updating as hardening, but note F35+
adds capsule protection so rollback may be blocked, and flashing resets all settings.

**Not blocking.**

- Batch 8→9 MMVQ/MMQ cliff (§7) — matters for parallel server slots.
- Widening weight loads in `vecdotq.cuh` (§4) — the only untried kernel lever, high blast radius.
- Upstream RDNA2 tile-kernel faults on ALiBi / attention sinks / exotic KV dtypes (§3).
- `ggml_cuda_mul_mat_vec_q` aborts at nwarps=2, `ncols_dst==1` — latent upstream bug, worth filing.

---

## 10. Verification

```bash
# correctness gates
./build/bin/test-backend-ops -o MUL_MAT          # expect 1297/1297
./build/bin/test-backend-ops -o FLASH_ATTN_EXT

# coherence — never trust a kernel speedup without this
./build/bin/llama-cli -m <model> -p "The capital of France is" -n 20

# perf
./build/bin/llama-bench -m <model> -fa 1 -p 512 -n 128
```

Kill switches: `GGML_CUDA_NO_Q8_1_CACHE=1`, `LLAMA_SPEC_NO_ADAPTIVE=1`,
`GGML_CUDA_DISABLE_GRAPHS=1` (also required for `rocprofv3 --kernel-trace` on the decode
path, which segfaults otherwise).

---

## 11. 2026-09-27 overnight: single-V620 speculative decoding push

Goal: the best experience on **one** V620, since that is what most people running this will
have. All numbers below are from a **warm llama-server** (every prompt run once untimed
first): a fresh process pays one-time costs on its first request (lazy kernel loading, first
CUDA-graph captures), e.g. DFlash math 68.5 t/s cold vs 79.3 warm with identical tokens.
Mean over four prompts (math, code, list, essay), 256 tokens, greedy unless noted.

### Results

| model | mode | start of night | final |
|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_XL | plain decode | 23.8 | 24.5 |
| | MTP n=3 (reduced-vocab drafter) | 49.6 | **54.2** |
| | **DFlash2, adaptive cap 7 (reduced-vocab drafter)** | 58.0 | **61.1** |
| Swift-Qwen3.8-27B | plain decode | 23.4 (their Q4_K_M) | 24.4 (our Q4_K_XL) |
| | MTP n=3 (reduced-vocab drafter) | 48.6 | 52.6 |
| | **DFlash2 n=3 (reduced-vocab drafter)** | 51.4 | **54.3** |
| | same, temp 1.0 / top-p 0.95 / top-k 20 (model card) | | 52.0 |

Final numbers are on commit 0b1887762. At ~8k tokens of context Swift + DFlash2 still runs at
45.9 t/s (plain 22.6).

### What landed

- **Reduced draft vocabulary** (`scripts/draft-vocab/build-draft-vocab.py`, MTP support in
  `qwen35.cpp`). The drafter's LM head was half its cost (3.0 of 6.2 ms per DFlash iteration;
  MTP runs it every draft step). The script copies the 64k most likely tokens' rows of the
  *target's own* output projection as raw quantized bytes, plus a `d2t` map; logits for kept
  tokens are bit-identical and output is unchanged. 98.9% coverage of held-out model output.
  +2.9% DFlash, +8.5% MTP (vs the stock drafter).
- **CUDA graph cache keyed by shape.** Batch sizes that differ only in token count shared one
  cache entry, so adaptive draft lengths kept evicting it: captures 226 -> 87 per session, +1.5%.
- **Fewer kernels per delta-net layer**: RMS_NORM+SCALE fusion (q/k L2 norm), the `ssm_out`
  and alpha-gate bias adds moved so they fuse into their matmuls, the batch-1 conv-input copy
  replaced by a reshape, and GATED_DELTA_NET reading its state straight from the cache instead
  of a gathered 3 MB copy (single-sequence batches). 1800 -> ~1520 dispatches per token,
  decode 23.90 -> 24.67; the 4-token verify 82.1 -> 83.2.
- **Flash-attn tile fixes scoped to RDNA2.** RDNA3/4 also use the tile kernel for decode and
  small batches; the shared table is upstream's again.

### Swift-Qwen3.8-27B quantization

Our build from Swift's F16: Unsloth's UD-Q4_K_XL per-tensor recipe with the IQ tensors moved to
Q4_K, Unsloth's imatrix (computed on the base model). KL divergence vs Swift's Q8_0, wikitext,
512 ctx, 95 chunks:

| | size | mean KLD | 99% KLD | same top token |
|---|---|---|---|---|
| Swift's own Q4_K_M | 16.79 GiB | 0.0134 | 0.144 | 95.27% |
| our Q4_K_XL (no IQ) | 16.57 GiB | **0.0092** | **0.093** | **96.41%** |

Swift's MTP head is the stock head (all F32 norms bit-identical to the base model), and unlike
Qwopus, DFlash2 transfers to it.

### Tried and rejected

- **MMVQ fusion for 2-8 columns** (residual add, gate+up GLU in the verify batch): the fused
  kernel variant is 5x slower at 8 columns (pp8 119 -> 23 t/s), even with only a bias.
- **Measured-throughput draft-length controller** (per-length moving averages of tokens/ms,
  periodic neighbour probes): worse than the existing one. A length's first use includes graph
  capture (up to 195 ms), which poisons its estimate; iteration cost is nearly flat for n=1..3
  so the choice rides on noisy, content-dependent token counts.
- Graphs off, fewer CPU threads, GPU-side target sampling: all within 1%.
- DFlash2's block width changes the accuracy of *every* drafted position (Swift code:
  first-token acceptance 0.92 at n=3, 0.84 at n=4), which is why n=3 often beats wider drafts.

### Where the time goes now

A DFlash n=3 iteration on Swift is ~56 ms: ~49 ms target verify of 4 tokens, 4.4 ms drafter,
~2-3 ms host. The verify runs at 88-93% of memory bandwidth; what remains is small kernels
(each <1%) or fewer bytes (a quality trade).

## 12. 2026-09-27 follow-up: the three remaining items

| model | mode | before | after |
|---|---|---|---|
| Qwen3.8-27B | DFlash2 adaptive cap 7 | 61.1 | **62.2** |
| | MTP n=3 | 54.2 | **55.0** |
| Swift-Qwen3.8-27B (our Q4_K_XL) | DFlash2, greedy | 54.3 (fixed n=3) | **58.5** (adaptive cap 7) |
| | DFlash2, temp 1.0 | 52.0 | **52.9** |

Commit 5f0437f39, warm llama-server, mean of four prompts.

- **Kernel merges in the verify batch (landed).** ADD->RMS_NORM->MUL (the residual add and the
  next pre-norm, writing both outputs; the allocator aliases the sum over one input and the
  result over the other, which the kernel handles) and ADD->SOFTPLUS->MUL with row-broadcast
  vectors (delta-net decay gate). The 4-token eval runs no binary-op kernels any more
  (224 -> 0), 2026 -> 1802 dispatches, pp4 83.3 -> 84.9.
- **Draft-length control (no new code needed).** Re-measured on the deployed setup, the
  existing adaptive controller now matches the best fixed length on every prompt for Swift
  too: the reduced-vocab drafter made longer drafts cheap. Launcher switched from fixed n=3 to
  cap 7.
- **Host gap between steps (stopped).** Per iteration: target submit 0.65 ms, drafter submit
  0.6, feature hand-off 0.2 (copy 0.02), sampling 0.5-0.7; the drafter's GPU time is ~4 ms.
  Outside GPU waits the main thread's work is spread over many functions at <0.5% each.
  The drafter context never reuses its graph (it alternates the injection and draft graphs),
  but that shows as ~0.1% of CPU samples. No single fixable cause.

## 13. 2026-09-27: speculative sampling for temperature 1.0 chat

With exact-match verification, a draft token survives only when the target's own random sample
equals it, so at temperature 1.0 the acceptance of a draft is p(draft). Commit 2055e58f9 adds
standard speculative sampling for DFlash2 (`--spec-draft-temp T`, server only, default off):
each draft position is sampled from softmax(selector scores / T) over its 16 candidates, and
verification accepts with probability min(1, p/q), otherwise emits a sample of
norm(max(0, p - q)) and stops. The output distribution is exactly the target's. Greedy
requests keep argmax drafts. The target distribution is taken with the grammar applied first,
so tool-call JSON (the web UI's MCP tools attach a lazy grammar) also goes through p/q
verification; the first version fell back to exact match there, which disabled the feature for
every answer once thinking ended.

Chat benchmark (Swift Q4_K_XL, the web UI system prompt, 4 research questions x 3 seeds,
temp 1.0 / top-p 0.95 / top-k 20, 700 tokens), interleaved:

| drafts | t/s | draft accepted |
|---|---|---|
| argmax (before) | 37.4, 37.4, 37.4 | 43% |
| sampled, T = 1.0 | 42.3, 42.6, 42.7, 43.5 | 49% |

Per verified position, sum min(p, q) = 0.67 against p(argmax q) = 0.58. Scored on the same
positions, drafter temperatures 0.85-1.0 are the flat optimum (0.4: 0.64, 1.3: 0.66), so the
drafter is well calibrated and T = 1.0 is used.

Checks: the accept/residual step against synthetic p and q (1M-4M trials, output frequencies
match p); 3000 short completions plain vs speculative, tokens 2-4 not distinguishable
(chi-square p 0.54-0.94); greedy text and PPL identical to the previous build. Both launchers
now pass `--spec-draft-temp 1.0`; greedy requests to such a server give byte-identical output.
With tools attached: 198 and 121 positions verified by p/q, none by exact match, tool call parsed.
Not exercised: the checkpoint-replay branch (needs a rollback beyond n_rs_seq = n_max, which
cannot happen here). MTP still drafts greedily and is the next candidate.

### 13a. The same for MTP (529c6fb90)

The MTP drafter samples each step from softmax(logits / T) over its top-10 candidates and feeds
the sampled token to the next step. Swift Q4_K_XL, MTP head with the 64k reduced vocabulary,
all at temp 1.0, interleaved:

| config | chat bench (t/s) | math / code / list / essay (t/s, mean of 2) | mean |
|---|---|---|---|
| MTP n=3, argmax drafts | 38.0, 37.9 (37% accepted) | | |
| MTP n=3, sampled | 44.2, 44.2 (48% accepted) | 62.1 / 54.1 / 64.3 / 42.7 | 55.8 |
| MTP n=4, sampled | | 64.5 / 56.0 / 70.9 / 38.4 | 57.4 |
| DFlash2 adaptive cap 7, sampled | 41.8, 42.9 (48-49% accepted) | 76.1 / 54.1 / 68.6 / 39.7 | 59.6 |

Per verified position, MTP: sum min(p,q) 0.66 vs p(argmax q) 0.57. Greedy output with the flag
on is identical; with tools attached no position falls back to exact match. With both
drafters sampling, MTP n=3 leads on research chat by ~4% and DFlash2 leads on the mixed prompts
by ~7% (math most of all).

## 14. 2026-09-27: flash-attention occupancy on RDNA2, D=256 retune (b96c5233d)

HIP reports 32768 registers per multiprocessor on gfx1030, a quarter of a WGP's register file,
so its occupancy query returned 0 for flash-attention tile kernels above 128 VGPRs (every larger
tile config aborted) and too few blocks for the rest (the small-batch kernels split the KV cache
too coarsely). launch_fattn now computes RDNA2 occupancy from the kernel's registers and LDS, and
the D=256 tile uses 64-key batches.

Per attention call, Qwen3.5 shape (24 Q heads over 4 KV heads, D=256):

| | before | after |
|---|---|---|
| prefill, 512 tokens | 12.0 TFLOPS | 15.7 TFLOPS |
| decode, 1 token, 64k context | 1200 us | 580 us |
| verify, 4 tokens, 64k | 1766 us | 678 us |
| verify, 8 tokens, 64k | 2190 us | 1162 us |

Swift-Qwen3.8-27B Q4_K_XL at 64k context (llama-bench): pp512 216.7 -> 245.8, tg32 16.5 -> 19.8.
Empty context unchanged (pp512 425). Web UI server, DFlash, one long document at growing depth:

| depth | prompt t/s before -> after | generation t/s before -> after |
|---|---|---|
| 16k | 366 -> 369 | 45.2 -> 49.3 |
| 32k | 304 -> 323 | 41.7 -> 45.5 |
| 64k | 243 -> 269 | 32.0 -> 37.0 |
| 96k | 191 -> 219 | 29.6 -> 41.9 (acceptance 55% -> 61%) |
| 128k | 158 -> 185 | 25.6 -> 38.4 (acceptance 56% -> 63%) |

Prompt speed is the newest 32k chunk. The 96k/128k generation gains are partly higher draft
acceptance in those runs; the controlled figure is the llama-bench one above.
Flash-attention tests pass for D=64/128/256/512/576; KLD against the flash-attention-off path
improved (mean 0.0063 -> 0.0056, max 1.47 -> 0.28). The server now runs 128k context.

Tried and rejected on the way: forced rocBLAS matmuls for prefill (371-378 vs 434 t/s), flash
attention off (slower at depth), ubatch 1024/2048 (no change), a 64-column tile for D=256 (slower).

## 15. 2026-09-27: follow-ups from an outside review

A read-only review of this work proposed ranked ideas; four were tried.

| idea | result |
|---|---|
| A. Run the 8 conv-state snapshot copies per delta-net layer (384 kernels per verify step at draft cap 7) as one launch (7de0c83d2) | +1.4% spec decode (59.3 -> 60.1 t/s, 4 prompts x 2 rounds, every prompt faster). Greedy output byte-identical; a mutant that writes the wrong slots changes it. |
| B1. Verify-batch attention configs: the 3-4 token entry compiled to 256 VGPRs with spills (abef34ca5) | 118 VGPRs, no spills; 8-token verify at 64k 1161 -> 1062 us per call, 4-token 679 -> 662 us. |
| B2. All 6 query heads of a KV head in one block (ncols2 = 6) | Correct, but every variant spills 23-333 VGPRs to scratch: prefill attention 15.5 -> 3.0 TFLOPS. The tile kernel does not handle a non-power-of-two group size efficiently; would need kernel rework. Not committed. |
| E. Truncate the drafter's distribution with the request's top-p before sampling | Scored on the same 7000 positions: expected acceptance 0.6654 -> 0.6687 at top-p 0.95 (worse at 0.8). About +0.5%; not implemented. |
| C. MMQ stream-k with more persistent blocks | The old stream-k test was starved (36 blocks, 1 per WGP): 302 t/s at 1x, 415.5 at 2x, but tiling is still faster (428.9). Closed. |

Also found: at temperature 1.0 the same request with the same seed does not give the same text
from run to run, even on an unchanged build (greedy does). Output comparisons must use greedy.
