import { app } from "../../../scripts/app.js";
import { addValueControlWidget } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "soze.csvreader",
    async nodeCreated(node) {
        if (node.comfyClass === "csvreader") {
            const rowInput = node.widgets.find(w => w.name === "row_number");
            if (rowInput) {
                const rownumbercontrol = addValueControlWidget(node, rowInput, rowInput.value + 1);
                node.widgets.splice(variationSeedWidgetIndex+1,0,node.widgets.pop());
            }
        }
    },
    // async beforeRegisterNodeDef(nodeType, nodeData, app) {
    //     if (nodeData.name === "csvreader") {
    //         const onExecuted = nodeType.prototype.onExecuted;
    //         nodeType.prototype.onExecuted = function(message) {
    //             onExecuted?.apply(this, arguments);
    //             if (this.widgets) {
    //                 const rowInput = this.widgets.find(w => w.name === "row_number");
    //                 if (rowInput && message.output && message.output.length > 11) {
    //                     rowInput.value = message.output[11];  // Set to the next row number
    //                     app.graph.setDirtyCanvas(true);
    //                 }
    //             }
    //         };
    //     }
    // }
});