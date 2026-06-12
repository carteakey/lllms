# archive

Scripts and units retired during the llama-swap migration.

## `setup-gemma-vision-service.sh`

Installer/controller for the old `gemma-vision.service` user unit (which
exec'd `run-models/run-llama-cpp-gemma-4-26b-a4b-vision.sh`). That run script
is gone. Serving is now handled by `llama-swap.service`, with
`gemma-4-26b-a4b-vision` as the preloaded default (see `llama-swap.yaml`).

If you previously installed the service, disable it before starting
llama-swap to avoid port conflicts:

```bash
systemctl --user disable --now gemma-vision.service
rm -f ~/.config/systemd/user/gemma-vision.service
systemctl --user daemon-reload
```
