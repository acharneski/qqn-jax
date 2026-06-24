#!/usr/bin/env node
'use strict';

/*
 * run_reports.js
 *
 * Node.js replacement for run_reports.sh.
 *
 * Runs example benchmarks with timestamped log files and supports
 * named "variants" — predefined parameter sets (environment variables
 * and/or CLI args) for running the examples in standard configurations.
 *
 * Usage:
 *   node run_reports.js                      # run the default variant set
 *   node run_reports.js --list              # list all available variants
 *   node run_reports.js fashion_default     # run one or more named variants
 *   node run_reports.js fashion_mnist fashion_relu_deep
 *   node run_reports.js --all               # run every defined variant
 *   node run_reports.js --report fashion_mnist_mlp_comparison
 *                                           # run all variants of one report
 */

const {spawn} = require('child_process');
const fs = require('fs');
const path = require('path');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function timestamp() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return (
        `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_` +
        `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`
    );
}

function ensureDir(dir) {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, {recursive: true});
    }
}

// ---------------------------------------------------------------------------
// Variant definitions
//
// Each variant has:
//   report : the example module name under ./examples/<report>.py
//   env    : extra environment variables to set for the run
//   args   : extra CLI args to pass to the script
//   desc   : human-readable description
// ---------------------------------------------------------------------------

const REPORTS_DIR = './examples';
const RESULTS_DIR = 'results';

const VARIANTS = {
    // -------------------- fashion_mnist_mlp_comparison --------------------
    fashion_default: {
        report: 'fashion_mnist_mlp_comparison',
        env: {},
        args: [],
        desc: 'Headline experiment: Fashion-MNIST, 256x3, tanh,gelu.',
    },
    fashion_mnist: {
        report: 'fashion_mnist_mlp_comparison',
        env: {DATASET: 'mnist'},
        args: [],
        desc: 'MLP comparison on plain MNIST.',
    },
    fashion_relu_deep: {
        report: 'fashion_mnist_mlp_comparison',
        env: {DEPTH: '5', HIDDEN: '128', ACTIVATION: 'relu'},
        args: [],
        desc: 'Deeper, narrower ReLU network (depth 5 x width 128).',
    },
    fashion_tapering: {
        report: 'fashion_mnist_mlp_comparison',
        env: {HIDDEN_SIZES: '256,128,64', ACTIVATION: 'tanh,gelu,gaussian'},
        args: [],
        desc: 'Tapering topology with mixed activations.',
    },
    fashion_lowvram: {
        report: 'fashion_mnist_mlp_comparison',
        env: {N_TRAIN: '8000', N_TEST: '2000', HIDDEN: '128', DEPTH: '2'},
        args: [],
        desc: 'Smaller problem sized for a low-VRAM GPU.',
    },
    fashion_qqn_deep_hessian: {
        report: 'fashion_mnist_mlp_comparison',
        env: {
            DATASET: 'fashion_mnist',
            N_TRAIN: '25000',
            N_TEST: '5000',
            HIDDEN: '256',
            DEPTH: '4',
            ACTIVATION: 'tanh,gelu',
        },
        args: [],
        desc:
            'Richer/more anisotropic Hessian (256x4) where the deep-memory ' +
            'curvature lever stays monotone — QQN`s strongest regime.',
    },
    fashion_qqn_wide: {
        report: 'fashion_mnist_mlp_comparison',
        env: {
            DATASET: 'fashion_mnist',
            N_TRAIN: '25000',
            N_TEST: '5000',
            HIDDEN: '512',
            DEPTH: '3',
            ACTIVATION: 'tanh,gelu,tanh',
        },
        args: [],
        desc:
            'Wider network (512x3): even richer curvature, amplifying the ' +
            'second-order advantage of QQN`s L-BFGS oracle.',
    },
    fashion_alt_linear: {
        report: 'fashion_mnist_mlp_comparison',
        env: {
            DATASET: 'fashion_mnist',
            N_TRAIN: '25000',
            N_TEST: '5000',
            HIDDEN: '128',
            DEPTH: '2',
            ACTIVATION: 'identity',
            F_TARGET: '0.35',
        },
        args: [],
        desc: 'Linear hidden layers (convex) where first-order accelerators (Anderson, Momentum) should excel.',
    },
    // ---- Profiling-enabled variants -------------------------------------
    // These mirror the headline config but switch on the integrated
    // profilers (JAX Profiler API + Perfetto traces, and Scalene hints).
    fashion_profile_jax: {
        report: 'fashion_mnist_mlp_comparison',
        env: {
            DATASET: 'fashion_mnist',
            N_TRAIN: '8000',
            N_TEST: '2000',
            HIDDEN: '128',
            DEPTH: '2',
            ACTIVATION: 'tanh,gelu',
            PROFILE: 'jax,perfetto',
            PROFILE_DIR: 'profiles',
            PROFILE_NAME: 'fashion_jax',
        },
        args: [],
        desc:
            'JAX Profiler + Perfetto trace capture on a small config. ' +
            'Load profiles/** in ui.perfetto.dev or TensorBoard Trace Viewer.',
    },
    fashion_profile_simple_fast: {
        report: 'fashion_mnist_mlp_comparison',
        env: {
            // DATASET: 'fashion_mnist',
            DATASET: 'mnist',
            N_TRAIN: '8000',
            N_TEST: '2000',
            HIDDEN: '64',
            DEPTH: '2',
            N_CLASSES: '10',
            ACTIVATION: 'sine',
            PROFILE: 'jax,perfetto',
            PROFILE_DIR: 'profiles',
            PROFILE_NAME: 'simple_fast',
            TIME_BUDGET: '15',  // seconds
            F_TARGET: '0.015',
        },
        args: [],
        desc: 'Fast, Small.',
    },
    fashion_profile_scalene: {
        report: 'fashion_mnist_mlp_comparison',
        env: {
            DATASET: 'fashion_mnist',
            N_TRAIN: '8000',
            N_TEST: '2000',
            HIDDEN: '128',
            DEPTH: '2',
            ACTIVATION: 'relu',
            PROFILE: 'scalene',
        },
        args: [],
        scalene: true,   // invoke via `scalene <script>` instead of `python3 <script>`
        desc:
            'Runs under Scalene for CPU+GPU+memory profiling. ' +
            'Outputs a live summary to the terminal.',
    },
    fashion_profile_all: {
        report: 'fashion_mnist_mlp_comparison',
        env: {
            DATASET: 'fashion_mnist',
            N_TRAIN: '8000',
            N_TEST: '2000',
            HIDDEN: '128',
            DEPTH: '2',
            ACTIVATION: 'tanh,gelu',
            PROFILE: 'all',
            PROFILE_DIR: 'profiles',
            PROFILE_NAME: 'fashion_all',
        },
        args: [],
        scalene: ['--html', '--outfile', 'profiles/scalene_fashion_all.html'],
        desc: 'Enable every profiling backend (JAX/Perfetto + Scalene hint).',
    },
    fashion_profile_scalene_html: {
        report: 'fashion_mnist_mlp_comparison',
        env: {
            DATASET: 'fashion_mnist',
            N_TRAIN: '8000',
            N_TEST: '2000',
            HIDDEN: '128',
            DEPTH: '2',
            ACTIVATION: 'tanh,gelu',
        },
        args: [],
        scalene: ['--html', '--outfile', 'profiles/scalene_fashion.html'],
        desc:
            'Scalene HTML report saved to profiles/scalene_fashion.html. ' +
            'Open in a browser for an interactive CPU+GPU+memory breakdown.',
    },


    // ----------------------- mnist_comparison ----------------------------
    mnist_default: {
        report: 'mnist_comparison',
        env: {},
        args: [],
        desc: 'Softmax MNIST optimizer comparison (default).',
    },

    // -------------------- mnist_sparse_benchmark -------------------------
    sparse_default: {
        report: 'mnist_sparse_benchmark',
        env: {},
        args: [],
        desc: 'Sparse MNIST benchmark (OrthantRegion, default).',
    },
    sparse_fashion: {
        report: 'mnist_sparse_benchmark',
        env: {DATASET: 'fashion_mnist'},
        args: [],
        desc: 'Sparse benchmark on Fashion-MNIST (harder corpus).',
    },
    sparse_aggressive_l1: {
        report: 'mnist_sparse_benchmark',
        env: {L1_SCALE: '1e-3', HIDDEN: '128', DEPTH: '2'},
        args: [],
        desc:
            'Aggressive L1 pressure (1e-3) on a 128x2 net — pushes harder ' +
            'on sparsity; inspect the accuracy/sparsity Pareto trade-off.',
    },
    sparse_precision_8bit: {
        report: 'mnist_sparse_benchmark',
        env: {QBITS: '8', QUANT_SCALE: '1e-3'},
        args: [],
        desc:
            'Precision-optimized to an 8-bit grid with a stronger quant ' +
            'penalty — targets near-lossless 8-bit quantization (low ' +
            'quant_err).',
    },
    sparse_precision_2bit: {
        report: 'mnist_sparse_benchmark',
        env: {QBITS: '2', QUANT_SCALE: '1e-3', HIDDEN: '128', DEPTH: '2'},
        args: [],
        desc:
            'Extreme 2-bit precision target: stresses the quantization ' +
            'region/penalty where the grid is coarsest.',
    },
    sparse_relu_deep: {
        report: 'mnist_sparse_benchmark',
        env: {ACTIVATION: 'relu', DEPTH: '3', HIDDEN: '128'},
        args: [],
        desc:
            'Deeper ReLU network (128x3): He-init sparsity/precision ' +
            'behavior vs. the default tanh baseline.',
    },
    sparse_mixed_taper: {
        report: 'mnist_sparse_benchmark',
        env: {
            DATASET: 'fashion_mnist',
            HIDDEN_SIZES: '256,128,64',
            ACTIVATION: 'tanh,gelu,gaussian',
            L1_SCALE: '1e-4',
        },
        args: [],
        desc:
            'Tapering 256->128->64 net with mixed activations on ' +
            'Fashion-MNIST — richer model for sparsity + precision study.',
    },
    sparse_fast: {
        report: 'mnist_sparse_benchmark',
        env: {
            N_TRAIN: '4000',
            N_TEST: '1000',
            HIDDEN: '64',
            DEPTH: '1',
            MAXITER: '2000',
        },
        args: [],
        desc:
            'Fast smoke-test config (small data, short budget) for quickly ' +
            'validating the base+polish pipeline end-to-end.',
    },
};

// Default set of variants to run when none are specified.
const DEFAULT_VARIANTS = [
    // "sparse_default",
    // "sparse_fast",            // quick end-to-end smoke test
    "sparse_fashion",         // harder corpus
    // "sparse_aggressive_l1",   // sparsity trade-off study
    // "sparse_precision_8bit",  // near-lossless 8-bit precision
    // 'fashion_default',
    // 'fashion_qqn_deep_hessian', // Demonstrates value and the test runs fast
    // "fashion_profile_simple_fast",
    // 'fashion_qqn_wide', // Successfully shows a wider advantage for QQN
    // 'fashion_alt_linear', // Interesting since it shows contrast - deeper lbfgs history hurts. NEEDS STOP TUNING.
    // 'fashion_profile_scalene', // Segfault?
];

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

function runVariant(name, variant, ts) {
    return new Promise((resolve) => {
        // Use a per-variant timestamp so sequential runs are distinguishable
        // and never silently collide.
        const variantTs = timestamp();
        const logfile = path.join(
            RESULTS_DIR,
            `${variant.report}_${name}_${variantTs}.log`
        );
        const scriptPath = path.join(REPORTS_DIR, `${variant.report}.py`);
        if (!fs.existsSync(scriptPath)) {
            console.error(
                `!!! Script not found for variant "${name}": ${scriptPath}`
            );
            resolve(1);
            return;
        }


        console.log(`\n=== Running variant "${name}" (${variant.report}) ===`);
        console.log(`    ${variant.desc}`);
        if (Object.keys(variant.env).length) {
            console.log(`    env: ${JSON.stringify(variant.env)}`);
        }
        if (variant.args.length) {
            console.log(`    args: ${variant.args.join(' ')}`);
        }
        console.log(`    log: ${logfile}`);

        // Truncate ('w') rather than append: a fresh timestamped file per run
        // should never accumulate stale content.
        const logStream = fs.createWriteStream(logfile, {flags: 'w'});
        // If the variant requests scalene execution, use scalene as the
        // launcher so it actually captures a profile rather than just
        // printing a hint.
        let executable, spawnArgs;
        if (variant.scalene) {
            const profileArgs = variant.scalene === true ? [] : variant.scalene;
            executable = 'scalene';
            // Scalene >= 2.3.0 uses a subcommand-based CLI: `scalene run <script>`.
            // Use `--` to clearly separate Scalene options from the script's own args.
            spawnArgs = ['run', ...profileArgs, '--', scriptPath, ...variant.args];
        } else {
            executable = 'python3';
            spawnArgs = [scriptPath, ...variant.args];
        }
        const child = spawn(executable, spawnArgs, {
            env: {...process.env, ...variant.env},
        });
        // Write a reproducible header so each log is self-describing.
        const startedAt = Date.now();
        const envPairs = Object.entries(variant.env)
            .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
            .join(' ');
        logStream.write(
            `# variant: ${name}\n` +
            `# report:  ${variant.report}\n` +
            `# desc:    ${variant.desc}\n` +
            `# started: ${new Date(startedAt).toISOString()}\n` +
            `# command: ${envPairs ? envPairs + ' ' : ''}` +
            `${executable} ${spawnArgs.join(' ')}\n` +
            `${'-'.repeat(72)}\n`
        );


        // Tee stdout/stderr to both the console and the log file.
        child.stdout.on('data', (data) => {
            process.stdout.write(data);
            logStream.write(data);
        });
        child.stderr.on('data', (data) => {
            process.stderr.write(data);
            logStream.write(data);
        });

        child.on('close', (code) => {
            const elapsedS = ((Date.now() - startedAt) / 1000).toFixed(1);
            logStream.write(
                `\n${'-'.repeat(72)}\n` +
                `# exit code: ${code}  elapsed: ${elapsedS}s\n`
            );
            logStream.end();
            if (code !== 0) {
                console.error(
                    `!!! variant "${name}" exited with code ${code} (${elapsedS}s)`
                );
            } else {
                console.log(`=== Finished variant "${name}" (${elapsedS}s) ===`);
            }
            resolve(code);
        });

        child.on('error', (err) => {
            console.error(`!!! Failed to start variant "${name}": ${err.message}`);
            logStream.end();
            resolve(1);
        });
    });
}

function listVariants() {
    console.log('Available variants:\n');
    const names = Object.keys(VARIANTS);
    const width = Math.max(...names.map((n) => n.length));
    for (const name of names) {
        const v = VARIANTS[name];
        console.log(`  ${name.padEnd(width)}  [${v.report}]  ${v.desc}`);
    }
    console.log('\nDefault set:', DEFAULT_VARIANTS.join(', '));
}

function parseArgs(argv) {
    const opts = {list: false, all: false, report: null, variants: []};
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === '--list' || a === '-l') {
            opts.list = true;
        } else if (a === '--all' || a === '-a') {
            opts.all = true;
        } else if (a === '--report' || a === '-r') {
            opts.report = argv[++i];
        } else if (a === '--help' || a === '-h') {
            opts.help = true;
        } else {
            opts.variants.push(a);
        }
    }
    return opts;
}

function printHelp() {
    console.log(`run_reports.js — run example benchmarks with named variants.

Usage:
  node run_reports.js [variant ...]      Run named variant(s).
  node run_reports.js --all              Run every defined variant.
  node run_reports.js --report <name>    Run all variants for one report.
  node run_reports.js --list             List available variants.
  node run_reports.js --help             Show this help.

With no arguments, runs the default set: ${DEFAULT_VARIANTS.join(', ')}.
`);
}

async function main() {
    const opts = parseArgs(process.argv.slice(2));

    if (opts.help) {
        printHelp();
        return;
    }
    if (opts.list) {
        listVariants();
        return;
    }

    ensureDir(RESULTS_DIR);
    ensureDir('profiles');
    const ts = timestamp();

    let selected;
    if (opts.all) {
        selected = Object.keys(VARIANTS);
    } else if (opts.report) {
        selected = Object.keys(VARIANTS).filter(
            (n) => VARIANTS[n].report === opts.report
        );
        if (selected.length === 0) {
            const reports = [
                ...new Set(Object.values(VARIANTS).map((v) => v.report)),
            ].sort();
            console.error(`No variants found for report "${opts.report}".`);
            console.error(`Known reports: ${reports.join(', ')}`);
            process.exitCode = 1;
            return;
        }
    } else if (opts.variants.length) {
        selected = opts.variants;
    } else {
        selected = DEFAULT_VARIANTS;
    }

    // Validate selection.
    const unknown = selected.filter((n) => !VARIANTS[n]);
    if (unknown.length) {
        console.error(`Unknown variant(s): ${unknown.join(', ')}`);
        console.error('Use --list to see available variants.');
        process.exitCode = 1;
        return;
    }

    let failures = 0;
    for (const name of selected) {
        const code = await runVariant(name, VARIANTS[name], ts);
        if (code !== 0) failures++;
    }

    console.log(
        `\nAll done. ${selected.length} variant(s) run, ${failures} failure(s).`
    );
    if (failures > 0) process.exitCode = 1;
}

main();