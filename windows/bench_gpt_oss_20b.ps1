$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ServerExe  = Join-Path $ScriptRoot 'vendor\llama.cpp\build\bin\llama-bench.exe'

$ModelFile = 'D:\local-llm-env\models\ggml-org\gpt-oss-20b-GGUF\gpt-oss-20b-mxfp4.gguf'

# Row-major speedup
$Env:LLAMA_SET_ROWS = '1'

$Args = @(
    '-m',            $ModelFile,
    '-ot', '.ffn_.*_exps.=CPU',         # this model has 36 MOE blocks. You can adjust this to move some MOE to the GPU.
    '--n-gpu-layers','999',          # everything else on the GPU, about 8GB
    '-fa',                             # max context (128k), flash attention
    '--jinja'
)

Write-Host "→ Benchmarking llama-bench "
Start-Process -FilePath $ServerExe -ArgumentList $Args -NoNewWindow

