import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";


const findWidgetByName = (node, name) => {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

const findInputByName = (node, name) => {
    return node.inputs ? node.inputs.find((w) => w.name === name) : null;
};
const doesInputWithNameExist = (node, name) => {
    return node.inputs ? node.inputs.some((input) => input.name === name) : false;
};


app.registerExtension({
    name: "soze.loadimagesfromfolder",
    async setup() {
        // Your widget update logic here
    },
    // async afterProcessing(nodeType, nodeData, app) {
    //     if (nodeData.name === "Load Images From Folder") {
    //         // This code runs when your custom node is being registered
    //         const onNodeCreated = nodeType.prototype.onNodeCreated;
    //         nodeType.prototype.onNodeCreated = function() {
    //             onNodeCreated?.apply(this, arguments);

    //             // Add a button to update the widget
    //             const updateButton = this.addWidget("button", "Update Value", "update", () => {
    //                 const newValue = Math.floor(Math.random() * 101); // Random value between 0 and 100
    //                 this.widgets.find(w => w.name === "value").value = newValue;
    //                 this.serialize_widgets();
    //             });
    //         };
    //     }
    // }
});