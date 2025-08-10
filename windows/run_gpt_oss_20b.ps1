$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ServerExe  = Join-Path $ScriptRoot 'vendor\llama.cpp\build\bin\llama-server.exe'

$ModelFile = 'D:\local-llm-env\models\ggml-org\gpt-oss-20b-GGUF\gpt-oss-20b-mxfp4.gguf'

# Row-major speedup
$Env:LLAMA_SET_ROWS = '1'

$Args = @(
    '-m',            $ModelFile,
    '--threads', '-1',
    # '--cpu-moe',         # this model has 36 MOE blocks. You can adjust this to move some MOE to the GPU.
    '-ot', '.ffn_.*_exps.=CPU',         # this model has 36 MOE blocks. You can adjust this to move some MOE to the GPU.
    '--n-gpu-layers','999',          # everything else on the GPU, about 8GB
    # '-c',            '0',
    '--ctx-size',   '16384',
    '-fa',                             # max context (128k), flash attention
    '--jinja',
    '--reasoning-format', 'none',
    '--host',        '0.0.0.0',
    '--port',        '8502',
    '--temp',   '1.0',
    '--min-p',  '0.0',
    '--top-p',  '1.0',
    '--top-k',  '0.0'
)

Write-Host "→ Starting llama-server on http://localhost:8080 ..."
Start-Process -FilePath $ServerExe -ArgumentList $Args -NoNewWindow

