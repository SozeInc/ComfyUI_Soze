// Adds a "Copy Names to Compare Text" button to the four enum-switch loader
// nodes. Click it once after picking models in the dropdowns and every
// compare_text_N widget is populated with the corresponding model's base
// filename (no path, no extension). Existing compare_text_N values are
// overwritten only for slots whose model dropdown has a non-empty value.
//
// The button also copies a JSON array of the populated compare-text names to
// the clipboard, e.g. ["4x-UltraSharp","4x-AnimeSharp"] — handy for pasting
// into an enum_value source elsewhere in the graph.

import { app } from "../../../scripts/app.js";

const ENUM_SWITCH_NODES = new Set([
    "Checkpoint Enum Switch",
    "Upscale Model Enum Switch",
    "Lora Enum Switch",
    "Diffusion Model Enum Switch",
]);

const MAX_CASES = 10;

function baseName(name) {
    if (!name || typeof name !== "string") return "";
    // Strip any directory portion (handle both / and \).
    let s = name.split(/[\\/]/).pop();
    // Strip the file extension (last dot, but leave dotfiles alone).
    const dot = s.lastIndexOf(".");
    if (dot > 0) s = s.slice(0, dot);
    return s;
}

// Copy text to the clipboard. Must run synchronously inside the click handler
// so it stays within the user-gesture stack (required by execCommand and by
// the async clipboard API's focus check).
//
// We try the synchronous execCommand path FIRST because it works on insecure
// origins too (e.g. http://<lan-ip>:8188, where navigator.clipboard is
// undefined). The async Clipboard API is used as a best-effort enhancement.
function copyToClipboard(text) {
    const sync = fallbackCopy(text);

    try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            // Fire-and-forget; swallow rejection (e.g. document-not-focused).
            navigator.clipboard.writeText(text).catch(() => {});
            return true;
        }
    } catch (_) { /* ignore */ }

    return sync;
}

function fallbackCopy(text) {
    try {
        const ta = document.createElement("textarea");
        ta.value = text;
        // Keep it on-screen-ish but invisible; off-screen textareas are
        // sometimes ignored by execCommand on certain browsers.
        ta.style.position = "fixed";
        ta.style.top = "0";
        ta.style.left = "0";
        ta.style.width = "1px";
        ta.style.height = "1px";
        ta.style.padding = "0";
        ta.style.border = "none";
        ta.style.outline = "none";
        ta.style.boxShadow = "none";
        ta.style.background = "transparent";
        ta.style.opacity = "0";
        ta.setAttribute("readonly", "");
        document.body.appendChild(ta);

        const sel = document.getSelection();
        const savedRange = sel && sel.rangeCount > 0 ? sel.getRangeAt(0) : null;

        ta.focus();
        ta.select();
        ta.setSelectionRange(0, ta.value.length); // iOS/Safari

        let ok = false;
        try { ok = document.execCommand("copy"); } catch (_) { ok = false; }

        document.body.removeChild(ta);

        // Restore any prior selection we clobbered.
        if (savedRange && sel) {
            sel.removeAllRanges();
            sel.addRange(savedRange);
        }
        return ok;
    } catch (_) {
        return false;
    }
}

// Populate compare_text_N from model_N base names, and copy a JSON array of the
// populated names (with a leading "none" when allow_none is on) to the clipboard.
function copyNamesToCompareText(node, nodeData) {
    const findWidget = (n) => (node.widgets || []).find((w) => w.name === n);

    let copied = 0;
    let skipped = 0;
    const names = [];
    for (let i = 1; i <= MAX_CASES; i++) {
        const modelW = findWidget(`model_${i}`);
        const cmpW = findWidget(`compare_text_${i}`);
        if (!modelW || !cmpW) { skipped++; continue; }
        const v = baseName(modelW.value);
        if (!v) { skipped++; continue; }
        cmpW.value = v;
        if (typeof cmpW.callback === "function") {
            try { cmpW.callback(v); } catch (_) { /* noop */ }
        }
        names.push(v);
        copied++;
    }

    if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);

    const allowNoneW = findWidget("allow_none");
    const clipNames = (allowNoneW && allowNoneW.value === true) ? ["none", ...names] : names;

    const jsonArray = JSON.stringify(clipNames);
    const clipOk = copyToClipboard(jsonArray);

    console.log(
        `[Soze] ${nodeData?.name}: copied ${copied} model name(s) to compare_text ` +
        `widgets (${skipped} skipped). Clipboard ${clipOk ? "OK" : "FAILED"}: ${jsonArray}`
    );
}

app.registerExtension({
    name: "soze.enum_switch_copy_names",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!ENUM_SWITCH_NODES.has(nodeData?.name)) return;

        // --- Robust value persistence (name-based) ---------------------------
        // ComfyUI saves widget values as a flat positional array
        // (`widgets_values`) and restores them by index. With many optional
        // widgets (incl. blank-default combos) the index at load can drift and
        // shift every value. We additionally persist a {name: value} map and
        // restore by name, re-applying on a deferred tick so we win over any
        // later positional pass. The map is stored in BOTH the serialized node
        // object and node.properties (the latter is always round-tripped).
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            onSerialize?.apply(this, arguments);
            const map = {};
            for (const w of (this.widgets || [])) {
                if (w && w.name && w.type !== "button") map[w.name] = w.value;
            }
            o.soze_widget_values = map;
            if (!this.properties) this.properties = {};
            this.properties.soze_widget_values = map;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (o) {
            onConfigure?.apply(this, arguments);
            const map = (o && o.soze_widget_values) ||
                (this.properties && this.properties.soze_widget_values);
            if (!map) return;
            const apply = () => {
                for (const w of (this.widgets || [])) {
                    if (w && w.name && w.type !== "button" &&
                        Object.prototype.hasOwnProperty.call(map, w.name)) {
                        w.value = map[w.name];
                    }
                }
                if (typeof this.setDirtyCanvas === "function") this.setDirtyCanvas(true, true);
            };
            apply();
            setTimeout(apply, 0);  // re-apply after any later positional restore
        };

        // Add the "Copy Names to Compare Text" button back, but guarantee it is
        // ALWAYS the last widget. The shift bug happened because the button sat
        // AHEAD of input widgets in this.widgets while being skipped in the
        // saved widgets_values array, so positional restore landed every value
        // one slot early. A trailing button is harmless: input widgets keep
        // indices 0..N-1 whether or not the button is serialized.
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const button = this.addWidget(
                "button",
                "Copy Names to Compare Text",
                null,
                () => copyNamesToCompareText(this, nodeData),
            );
            // Belt: ask ComfyUI not to serialize it at all.
            button.serialize = false;
            if (!button.options) button.options = {};
            button.options.serialize = false;

            // Suspenders: keep it pinned to the end of the widget list, even if
            // ComfyUI created the input widgets after onNodeCreated ran.
            const moveButtonLast = () => {
                if (!this.widgets) return;
                const i = this.widgets.indexOf(button);
                if (i !== -1 && i !== this.widgets.length - 1) {
                    this.widgets.splice(i, 1);
                    this.widgets.push(button);
                }
            };
            moveButtonLast();
            setTimeout(moveButtonLast, 0);

            return r;
        };

        // Also expose the same action on the node's right-click menu (adds no
        // widget; handy and harmless alongside the button).
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            const r = getExtraMenuOptions ? getExtraMenuOptions.apply(this, arguments) : undefined;
            options.unshift(
                {
                    content: "Copy Names to Compare Text",
                    callback: () => copyNamesToCompareText(this, nodeData),
                },
                null, // separator
            );
            return r;
        };
    },
});
