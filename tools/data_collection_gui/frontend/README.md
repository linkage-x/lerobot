# LeRobot Data Collection GUI

Initial web skeleton for handheld recording, Rerun-style trajectory review, and real robot trajectory replay.

```bash
cd tools/data_collection_gui/frontend
npm install
npm run dev
```

The current build uses a mock API adapter with the same command surface expected from a future local Python gateway.

To use the local gateway contract:

```bash
python -m tools.data_collection_gui.gateway \
  --config-path tools/handheld/handheld_record_example.yaml \
  --datasets-root outputs/datasets \
  --port 8765
```

Vite proxies `/api/*` to `http://127.0.0.1:8765` by default. If the gateway is on a different port (e.g. when 8765 is held by VS Code), point Vite at it via `GUI_API_TARGET`:

```bash
GUI_API_TARGET=http://127.0.0.1:8766 npm run dev
```

If the proxy fails to reach the gateway, the frontend falls back to the mock adapter and pages like *Dataset Processing* will show "Gateway not connected" instead of real datasets.
