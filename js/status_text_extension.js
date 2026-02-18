// Status text display extension for ComfyUI custom nodes
// Adds a status area at the bottom of nodes and a simpler single-line variant.

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "custom_nodes.status_text_display",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Limit this extension to the Soze ComfyDeploy 'Download Files' node only.
        // Accept multiple possible identifiers that might appear in nodeData.
        const isTargetNode = (
            nodeData?.name === "Soze_ComfyDeployDownloadAPIFiles" ||
            nodeData?.comfyClass === "ComfyDeploy API Download Files" ||
            (nodeData?.python_module && nodeData.python_module.includes("comfydeploy"))
        );
        if (!isTargetNode) {
            return;
        }

        // --- Full status box implementation ---
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            originalOnNodeCreated?.apply(this, arguments);
            this.statusText = "Ready";
            if (!this.size) this.size = [315, 200];
            else this.size[1] = Math.max(this.size[1] || 0, 200);
        };

        const originalOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            originalOnDrawForeground?.apply(this, arguments);

            // Draw the status text area at the bottom
            const padding = 10;
            const textAreaHeight = 50;
            const statusY = (this.size?.[1] || 200) - (textAreaHeight + padding);

            // Background
            ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
            ctx.fillRect(padding, statusY, (this.size?.[0] || 315) - (padding * 2), textAreaHeight);

            // Border
            ctx.strokeStyle = "rgba(100, 150, 200, 0.5)";
            ctx.lineWidth = 1;
            ctx.strokeRect(padding, statusY, (this.size?.[0] || 315) - (padding * 2), textAreaHeight);

            // Text
            const maxTextWidth = (this.size?.[0] || 315) - (padding * 4);
            ctx.fillStyle = "#aaa";
            ctx.font = "12px Arial";
            ctx.textAlign = "left";

            const lines = this.wrapText ? this.wrapText(this.statusText || "", maxTextWidth, ctx) : defaultWrapText(this.statusText || "", maxTextWidth, ctx);
            const lineHeight = 14;
            const maxLines = Math.floor((textAreaHeight - 10) / lineHeight);
            const visibleLines = lines.slice(0, maxLines);

            visibleLines.forEach((line, index) => {
                ctx.fillText(line, padding + 5, statusY + 15 + (index * lineHeight));
            });
        };

        nodeType.prototype.wrapText = function(text, maxWidth, ctx) {
            const words = String(text).split(" ");
            const lines = [];
            let currentLine = "";

            words.forEach(word => {
                const testLine = currentLine ? currentLine + " " + word : word;
                const metrics = ctx.measureText(testLine);
                if (metrics.width > maxWidth && currentLine) {
                    lines.push(currentLine);
                    currentLine = word;
                } else {
                    currentLine = testLine;
                }
            });
            if (currentLine) lines.push(currentLine);
            return lines;
        };

        nodeType.prototype.updateStatusText = function(text) {
            this.statusText = text;
            app.canvas?.setDirty?.(true);
        };

        // --- Simple single-line status implementation (kept lightweight) ---
        const simpleName = nodeData?.name?.toLowerCase?.() || "";
        if (simpleName.includes("simple-status")) return; // avoid duplicating

        const originalOnNodeCreatedSimple = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            originalOnNodeCreatedSimple?.apply(this, arguments);
            this._singleStatus = this._singleStatus || "";
        };

        const originalDrawSimple = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            originalDrawSimple?.apply(this, arguments);
            const y = (this.size?.[1] || 200) - 15;
            ctx.fillStyle = "#888";
            ctx.font = "11px monospace";
            ctx.fillText("Status: " + (this._singleStatus || ""), 10, y);
        };

        nodeType.prototype.setStatus = function(text) {
            this._singleStatus = text;
            app.canvas?.setDirty?.(true);
        };
    },

    async nodeExecuted(message, uniqueId, node, app) {
        if (!node) return;
        const statusText = message?.output?.ui?.status?.[0] || message?.output?.status || message?.output || "Execution Complete";
        if (node.updateStatusText) node.updateStatusText(statusText);
        if (node.setStatus) node.setStatus(typeof statusText === "string" ? statusText : JSON.stringify(statusText));
    },
});

function defaultWrapText(text, maxWidth, ctx) {
    const words = String(text).split(" ");
    const lines = [];
    let currentLine = "";
    words.forEach(word => {
        const testLine = currentLine ? currentLine + " " + word : word;
        const metrics = ctx.measureText(testLine);
        if (metrics.width > maxWidth && currentLine) {
            lines.push(currentLine);
            currentLine = word;
        } else {
            currentLine = testLine;
        }
    });
    if (currentLine) lines.push(currentLine);
    return lines;
}
