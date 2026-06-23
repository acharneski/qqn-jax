# `qqn_jax/profiling.py` — Profiling Integration

This module provides a lightweight, dependency-optional facade over three
complementary profiling backends. Any example or benchmark can opt into
profiling via environment variables without embedding profiler-specific
plumbing in application code.

---

## Backends

| Backend    | What it captures                          | View with                   |
|------------|-------------------------------------------|-----------------------------|
| `jax`      | Device + host op traces (`jax.profiler`)  | TensorBoard Trace Viewer    |
| `perfetto` | Same trace in Perfetto protobuf form      | <https://ui.perfetto.dev>   |
| `scalene`  | Whole-process CPU + GPU + memory sampling | `scalene` HTML / CLI report |

> **Note:** `jax` and `perfetto` share the same underlying trace. JAX's
> profiler writes a Perfetto-compatible `*.perfetto_trace.json.gz` file
> alongside the TensorBoard format, so selecting either backend (or both)
> starts the same `jax.profiler` session. They are kept independently
> selectable for clarity in the printed output.

---

## Environment Variables

| Variable       | Default          | Description                                                                                   |
|----------------|------------------|-----------------------------------------------------------------------------------------------|
| `PROFILE`      | *(unset)*        | Comma-separated backends to enable: `jax`, `perfetto`, `scalene`, or `all`. Empty = disabled. |
| `PROFILE_DIR`  | `./profiles`     | Output directory for emitted trace artifacts.                                                 |
| `PROFILE_NAME` | *(session name)* | Basename prefix for emitted artifacts (overrides the `name` argument to `profile_session`).   |

### `PROFILE` accepted values

| Value                                  | Effect                                     |
|----------------------------------------|--------------------------------------------|
| *(empty)*, `0`, `false`, `off`, `none` | All profiling disabled (zero overhead).    |
| `1`, `true`, `on`, `all`               | All three backends enabled.                |
| `jax`                                  | JAX/Perfetto trace only.                   |
| `perfetto`                             | JAX/Perfetto trace only (alias for `jax`). |
| `scalene`                              | Scalene annotation / hint only.            |
| `jax,perfetto,scalene`                 | All three backends (same as `all`).        |

---

## Quick Start

```bash
# Capture a Perfetto-loadable trace into ./profiles:
PROFILE=jax,perfetto PROFILE_DIR=profiles \
    python examples/fashion_mnist_mlp_comparison.py

# Whole-process CPU + GPU + memory profile (Scalene wraps the interpreter):
scalene examples/fashion_mnist_mlp_comparison.py

# All backends at once:
PROFILE=all python examples/fashion_mnist_mlp_comparison.py
```

In code:

```python
from qqn_jax.profiling import profile_session, profile_region

with profile_session("my_run"):  # whole-run device + host trace
    with profile_region("QQN-L80"):  # named span in the Perfetto timeline
        solver.run(x0)
```

Each optimizer variant in the comparison benchmark is automatically wrapped
in a `profile_region`, so the Perfetto timeline cleanly separates the work
done by each method.

---

## API Reference

### `profile_session(name="run")`

Context manager that wraps a whole benchmark run with the selected profilers.

**Behaviour by backend:**

- **`jax` / `perfetto`** — Calls `jax.profiler.start_trace(trace_dir,
  create_perfetto_trace=True)` on entry and `jax.profiler.stop_trace()` on
  exit. Traces are written to `PROFILE_DIR/<name>_<timestamp>/`. On exit,
  exact UI-load instructions are printed to stdout.
- **`scalene`** — Scalene must wrap the *whole* interpreter and cannot be
  toggled mid-process. If the process is already running under Scalene the
  session prints a confirmation; otherwise it prints the exact command to
  re-launch under Scalene (and a `pip install scalene` hint if the package
  is absent).

**When no backends are enabled** this is a zero-overhead no-op (no imports,
no allocations).

```python
@contextlib.contextmanager
def profile_session(name: str = "run") -> Iterator[None]: ...
```

| Parameter | Type  | Default | Description                                              |
|-----------|-------|---------|----------------------------------------------------------|
| `name`    | `str` | `"run"` | Basename prefix for the trace directory / artifact name. |

**Example:**

```python
with profile_session("rosenbrock_sweep"):
    for dim in [10, 100, 1000]:
        solver.run(jnp.zeros(dim))
```

---

### `profile_region(label)`

Context manager that annotates a named sub-range inside an active
JAX / Perfetto trace.

Uses `jax.profiler.TraceAnnotation` so the region appears as a named span
in the Perfetto timeline (e.g. one span per optimizer variant). This is a
no-op when profiling is disabled, when neither `jax` nor `perfetto` is in
`PROFILE`, or when JAX is unavailable. Exceptions from the annotation
machinery are silently swallowed so profiling can never break the actual
computation.

```python
@contextlib.contextmanager
def profile_region(label: str) -> Iterator[None]: ...
```

| Parameter | Type  | Description                                    |
|-----------|-------|------------------------------------------------|
| `label`   | `str` | Name shown in the Perfetto / TensorBoard span. |

**Example:**

```python
for name, solver in solvers.items():
    with profile_region(name):
        x_opt, state = solver.run(x0)
```

---

### `scalene_active() -> bool`

Returns `True` if the current process is running under Scalene.

Detection heuristic (either condition is sufficient):

1. The `SCALENE_PROFILE` environment variable is set (Scalene injects this).
2. `"scalene"` appears in `sys.modules` (Scalene has been imported).

```python
def scalene_active() -> bool: ...
```

**Example:**

```python
from qqn_jax.profiling import scalene_active

if scalene_active():
    print("Scalene is active — memory annotations will be captured.")
```

---

### `device_memory_report() -> str | None`

Returns a short human-readable JAX device memory summary, or `None` if
memory statistics are unavailable (older JAX, non-GPU backend, or any
exception).

Useful to log alongside Perfetto traces and Scalene reports so the
memory-pressure context is captured next to timing data.

```python
def device_memory_report() -> str | None: ...
```

**Output format** (one line per device with available stats):

```
TFRT_CPU_0: 0.12 / 8.00 GiB
cuda:0: 3.41 / 15.78 GiB
```

**Example:**

```python
from qqn_jax.profiling import device_memory_report

report = device_memory_report()
if report:
    print("[memory]", report)
```

---

## Trace Artifacts

When `jax` or `perfetto` is enabled, `profile_session` creates a
timestamped subdirectory under `PROFILE_DIR`:

```
profiles/
└── my_run_20240315-142301/
    ├── plugins/
    │   └── profile/
    │       └── ...          ← TensorBoard Trace Viewer format
    └── *.perfetto_trace.json.gz   ← Perfetto UI format
```

### Viewing traces

**Perfetto UI** (recommended for per-op and per-region inspection):

1. Open <https://ui.perfetto.dev> in Chrome / Edge.
2. Click **Open trace file** and select the `.perfetto_trace.json.gz` file.
3. Named `profile_region` spans appear as labelled rows in the timeline.

**TensorBoard Trace Viewer:**

```bash
tensorboard --logdir ./profiles
# Navigate to: Profile → Trace Viewer
```

---

## Integration with the Comparison Benchmark

The `examples/fashion_mnist_mlp_comparison.py` benchmark wraps each
optimizer variant in a `profile_region` automatically:

```python
from qqn_jax.profiling import profile_session, profile_region

with profile_session("mlp_comparison"):
    for variant_name, solver in variants.items():
        with profile_region(variant_name):
            x_opt, state = solver.run(x0)
```

This produces a Perfetto timeline where each optimizer's work is a
clearly-labelled, non-overlapping span, making it straightforward to
compare device utilisation and host overhead across methods.

---

## Dependency Notes

| Package    | Required? | Notes                                                                                                                                    |
|------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------|
| `jax`      | Soft      | Imported lazily inside `profile_session` / `profile_region`. If absent, those contexts silently no-op.                                   |
| `perfetto` | Optional  | The standalone `perfetto` Python package enables programmatic trace queries. JAX's own output is already Perfetto-compatible without it. |
| `scalene`  | Optional  | Must be installed (`pip install scalene`) and used to launch the script. The module prints install/launch hints if absent.               |

All imports are deferred to call time. Importing `qqn_jax.profiling` has
**zero overhead** when `PROFILE` is unset.

---

## Public API

```python
from qqn_jax.profiling import (
    profile_session,  # context manager — whole-run trace
    profile_region,  # context manager — named sub-range
    scalene_active,  # bool — is Scalene wrapping this process?
    device_memory_report,  # str | None — per-device memory summary
)
```