// Upload + drag-and-drop for the Soze "Load Audio" node.
//
// The built-in ComfyUI audio-upload widget (audio_upload / AUDIOUPLOAD) crashes
// for legacy INPUT_TYPES nodes because it assumes the new schema-created audio
// player widget exists. So the Python node uses a plain combo, and this
// extension adds the upload UX ourselves:
//   - an "Upload Audio" button (opens a file picker), and
//   - drag-and-drop of audio files onto the node.
// Both POST to ComfyUI's standard /upload/image input endpoint (which saves any
// uploaded file into the input directory) and then select it in the dropdown.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "Load Audio"; // NODE_CLASS_MAPPINGS key for Soze_LoadAudio
const AUDIO_EXTS = [".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus", ".aiff", ".aif"];

function isAudioFile(name) {
    const n = (name || "").toLowerCase();
    return AUDIO_EXTS.some((e) => n.endsWith(e));
}

function findAudioWidget(node) {
    return (node.widgets || []).find((w) => w.name === "audio");
}

// Upload one File to the ComfyUI input directory; returns the saved path
// (subfolder/name or name).
async function uploadToInput(file) {
    const body = new FormData();
    body.append("image", file, file.name); // route field is "image" even for non-images
    body.append("type", "input");
    body.append("overwrite", "true");

    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (resp.status !== 200) {
        const text = await resp.text().catch(() => "");
        throw new Error(`HTTP ${resp.status} ${text}`);
    }
    const data = await resp.json();
    return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

function selectInWidget(node, value) {
    const w = findAudioWidget(node);
    if (!w) return;
    if (!w.options) w.options = {};
    if (!Array.isArray(w.options.values)) w.options.values = [];
    // Drop the empty placeholder once we have a real file.
    const i = w.options.values.indexOf("");
    if (i !== -1 && value) w.options.values.splice(i, 1);
    if (!w.options.values.includes(value)) w.options.values.push(value);
    w.value = value;
    if (typeof w.callback === "function") {
        try { w.callback(value); } catch (_) { /* noop */ }
    }
    if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
}

async function uploadFilesAndSelect(node, files) {
    for (const f of files) {
        if (!isAudioFile(f.name)) continue;
        try {
            const path = await uploadToInput(f);
            selectInWidget(node, path);
            console.log(`[Soze] Load Audio: uploaded "${path}"`);
        } catch (err) {
            console.error("[Soze] Load Audio upload failed:", err);
            alert(`Audio upload failed: ${err.message}`);
        }
    }
}

app.registerExtension({
    name: "soze.load_audio_upload",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        // Upload button (hidden <input type=file> behind it).
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.accept = "audio/*";
            fileInput.multiple = false;
            fileInput.style.display = "none";
            fileInput.onchange = async () => {
                if (fileInput.files && fileInput.files.length) {
                    await uploadFilesAndSelect(this, [...fileInput.files]);
                }
                fileInput.value = "";
            };
            document.body.appendChild(fileInput);
            this._sozeAudioFileInput = fileInput;

            const button = this.addWidget("button", "Upload Audio", null, () => fileInput.click());
            button.serialize = false;
            if (!button.options) button.options = {};
            button.options.serialize = false;

            return r;
        };

        // Clean up the detached <input> when the node is removed.
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            try { this._sozeAudioFileInput?.remove(); } catch (_) { /* noop */ }
            return onRemoved ? onRemoved.apply(this, arguments) : undefined;
        };

        // Accept files dragged over the node.
        const onDragOver = nodeType.prototype.onDragOver;
        nodeType.prototype.onDragOver = function (e) {
            if (e?.dataTransfer?.items) {
                const hasFile = [...e.dataTransfer.items].some((it) => it.kind === "file");
                if (hasFile) return true;
            }
            return onDragOver ? onDragOver.apply(this, arguments) : false;
        };

        // Handle files dropped on the node. Accept synchronously (return true),
        // upload in the background.
        const onDragDrop = nodeType.prototype.onDragDrop;
        nodeType.prototype.onDragDrop = function (e) {
            const files = e?.dataTransfer?.files ? [...e.dataTransfer.files] : [];
            const audio = files.filter((f) => isAudioFile(f.name));
            if (audio.length) {
                uploadFilesAndSelect(this, audio);
                return true;
            }
            return onDragDrop ? onDragDrop.apply(this, arguments) : false;
        };
    },
});
