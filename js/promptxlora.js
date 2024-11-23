import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Soze_PromptXLora",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if(!nodeData?.name?.startsWith("Prompt X Lora")) {
			return;
		  }
          const onPromptXLoraConnectInput = nodeType.prototype.onConnectInput;
          nodeType.prototype.onConnectInput = function (targetSlot, type, output, originNode, originSlot) {
              const v = onPromptXLoraConnectInput? onPromptXLoraConnectInput.apply(this, arguments): undefined
              this.outputs[1]["name"] = "prompts_count"
              this.outputs[2]["name"] = "loras_count" 
              this.outputs[3]["name"] = "current_prompt_index"
              this.outputs[4]["name"] = "current_lora_index"
              return v;
          }
          const onPromptXLoraExecuted = nodeType.prototype.onExecuted;
          nodeType.prototype.onExecuted = function(message) {
              const r = onPromptXLoraExecuted? onPromptXLoraExecuted.apply(this,arguments): undefined
              let values = message["text"].toString().split('|').map(Number);
              this.outputs[1]["name"] = "(" + values[0] + ") - " + "prompts_count"
              this.outputs[2]["name"] = "(" + values[1] + ") - " + "loras_count" 
              this.outputs[3]["name"] = "(" + values[2] + ") - " + "current_prompt_index"
              this.outputs[4]["name"] = "(" + values[3] + ") - " + "current_lora_index"
              return r
          }
    }});