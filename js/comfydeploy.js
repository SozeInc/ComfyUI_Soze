import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "Soze.ComfyDeploy.Status",
    async nodeCreated(node) {
        if (node.comfyClass === "ComfyDeploy API Download Files") {
            console.log("✓ Found ComfyDeploy API Download Files node");
            
            // Listener that updates status and forwards to canvas-based status methods
            // (legacy widget removed — status is rendered by the canvas extension)
            
            // Listener that updates status
            const onStatus = (event) => {
                console.log("Status event received:", event.detail);
                try {
                    const { node_id, status_text } = event.detail;
                    console.log("Node ID match check:", node_id, "===", node.id.toString());
                    
                    if (node_id === node.id.toString()) {
                        console.log("✓ Status update for this node:", status_text);
                        // Forward status to the canvas-rendered status (if available)
                        try {
                            if (typeof node.updateStatusText === "function") node.updateStatusText(status_text);
                            if (typeof node.setStatus === "function") node.setStatus(status_text);
                        } catch (e) {
                            console.warn("Error forwarding status to node methods", e);
                        }
                    }
                } catch (e) {
                    console.warn("Status event handling error", e);
                }
            };

            // Register listener on the api event bus
            api.addEventListener("soze.comfydeploy.status", onStatus);
            console.log("✓ Event listener registered");

            // Cleanup when node is removed
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function() {
                try { api.removeEventListener("soze.comfydeploy.status", onStatus); } catch(e) {}
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };
        }
    }
});
