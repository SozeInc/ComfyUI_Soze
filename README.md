# ComfyUI_Soze

Quality-of-life custom nodes for ComfyUI: batch processing helpers, CSV-driven prompt iteration, file/folder loaders, string and JSON utilities, and integrations for ComfyDeploy, FAL, ElevenLabs, and Azure Blob Storage.

Originally bundled the now-abandoned [Comfy_KepListStuff](https://github.com/M1kep/Comfy_KepListStuff) helpers.

![workflow (1)](https://github.com/user-attachments/assets/a8f2869a-f678-49f7-ac78-4616fb1f9f43)

## Install

Clone into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/SozeInc/ComfyUI_Soze.git
pip install -r ComfyUI_Soze/requirements.txt
```

Or install via Comfy Registry / ComfyUI Manager.

## Configuration

Some nodes need credentials in environment variables (or a `.env` file in the repo root):

| Variable | Used by |
|---|---|
| `CD_API_KEY` | ComfyDeploy nodes |
| `ELEVENLABS_API_KEY` | ElevenLabs voice retriever |
| `SOZE_AZURE_STORAGE_CONNECTION_STRING` | ComfyDeploy image parameter upload |
| `FAL_KEY` (or `[API] FAL_KEY` in `config.ini`) | FAL nodes (e.g. Veo3.1) |
| `TENSORSCALE_API_KEY` | TensorScale nodes. Keys are model-scoped, so you can instead set one per model: `TENSORSCALE_API_KEY_MINIMAX_H3`, `_LTX2_5_FAST`, `_LTX_2_3_FAST`, `_COSMOS3_NANO`, `_HUNYUAN_IMAGE_3`, `_SENSENOVA_U1_5` (each falls back to `TENSORSCALE_API_KEY`). `TENSORSCALE_BASE_URL` overrides the host. |

The repo's `.gitignore` excludes `.env` and `config.ini`. **Never commit credentials.** If you suspect a key has leaked, rotate it.

## Node groups

- **CSV** — `CSV Reader`, `CSV Reader X Checkpoint`, `CSV Reader X Lora`, `CSV Writer`. The `X Checkpoint`/`X Lora` variants iterate `index` across rows × models, so a single counter walks every (row, checkpoint) pair.
- **Files / folders** — `Load Files From Folder`, `Load Files With Pattern`, `File Loader`, `Does File Exist`, `Save Image With Absolute Filename`, `Save Text File To Output`, `Append To Text File`.
- **Images** — `Load Image`, `Load Images From Folder`, `Load Images From Folder X Lora`, `Load Image From URL`, `Load Image From Filepath`, `Image Batch Process Switch`, `Image Overlay`, `Empty Images`, `Variable Image Builder`, `Shrink Image`, `Pad Mask`, `Get Most Common Image Colors`, `Multi Image Batch`, `Soze Image Size With Maximum`.
- **Video** — `Append To Video` (ffmpeg concat with audio), `Load Videos From Folder`.
- **Strings** — `Multiline Concatenate`, `Multi Find And Replace`, `Special Character Replacer`, `Text Contains`, `String Functions`, `String Splitter`, `Empty String Replacement`, `Prompt Cache`, `Output Filename`.
- **JSON** — `JSON Value Parser`, `JSON Path Extractor`, `JSON Array Iterator`, `JSON File Loader`, `JSON Formatter`, `JSON Get Array Count`, `Create Image Batch From JSON Array`, `Load Images From JSONArray`.
- **Range / XY** — `Int/Float Range (Step)`, `Int/Float Range (Num Steps)`, `XY Any`, `XY Image`.
- **Converters** — `Int/Float/Bool/String` cross-conversion nodes, `Boolean Inverter`.
- **Loaders** — `Lora File Loader` (load by absolute path), `Checkpoint File Loader` (load by absolute path).
- **Integrations** — ComfyDeploy API queue / cache / download nodes, ElevenLabs voice retriever, FAL Veo3.1 ref-img-to-video.
- **TensorScale** — synchronous world-model inference (no polling; each node POSTs once and streams the finished media back). `TensorScale MiniMax H3 Video` (`fl2va`) and `... Video Reference` (`ref2va`) for video+audio, `TensorScale LTX-2.5 Fast Video` (video+synced audio, $0.04/video second), `TensorScale LTX-2.3 Fast Video`, `TensorScale Cosmos3 Nano Text To Video` / `... Video To Video`, `TensorScale Hunyuan Image 3 Fast`, `TensorScale SenseNova U1.5`. Reference media is inlined as a base64 data URI by default; the whole request body is capped at 10 MiB, so use the `*_url` widgets or flip `use_fal_upload` on for anything video-sized. [API docs](https://www.tensorscale.io/docs.html)

## ffmpeg

`Append To Video` shells out to `ffmpeg`; it must be on `PATH`.

## Contributing

Issues and PRs welcome. All respect to the original creators of any subsumed nodes — happy to upstream changes if you'd prefer them in your repo.

## License

MIT — see [LICENSE](LICENSE).
