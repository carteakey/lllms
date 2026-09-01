# Benchmark presets

Benchmark presets are deliberately plain text and live next to the script
that owns them. A preset is a shell fragment containing only environment
assignments consumed by run-llama-bench.sh or run-ik-llama-bench.sh.

For example, fast.env can contain:

    N_GPU_LAYERS=99
    N_PROMPT=512
    N_GEN=128
    REPETITIONS=3

Load a preset explicitly before launching a script:

    set -a
    . bench-models/presets/fast.env
    set +a
    ./bench-models/bench-llama-cpp-gpt-oss-120b.sh

The launcher does not source arbitrary files. A future UI selector must parse
and validate assignments before adding them to a child process environment.
