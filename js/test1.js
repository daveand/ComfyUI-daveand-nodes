import { app } from "../../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { addValueControlWidget } from "../../../scripts/widgets.js";

app.registerExtension({
  name: "daveand.test1",
  async init(app) {},
  async setup(app) {},

  async onExecute(app) {
    console.log("Test1 node execute");

    this.onPropertyChanged = function(action, data)
    {
      console.log("Test1 node action", action, data);
      return true;
    }
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "Test1") {

      // Create node
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = async function () {
        const r = onNodeCreated
        
        // console.log("🚀 Test1 node created");
        // console.log(r);

        // this.onMouseDown = function( event, pos, graphcanvas )
        // {
        //   console.log("Test1 node mouse down", event, pos, graphcanvas);
        //   return true; //return true is the event was used by your node, to block other behaviours
        // }

        api.addEventListener("daveand_data_to_ui", async ({ detail }) => {
          const { item, index } = detail;
          console.log("Received data:", item, index);

          // const index_widget2 = addValueControlWidget(this, "index2", {
          //     type: "number",
          //     label: "Index2",
          //     default: 0,
          // });
          // index_widget2.setValue(index);

          // const index_widget = this.widgets?.findIndex((w) => w.name === 'index');
          // console.log("Index widget:", index_widget);


          //this.addInput("A","number");

          var w = this.widgets?.find((w) => w.name === 'index')
          if (w) {
              w.value = index;
              this.onResize?.(this.size);  // onResize redraws the node
          }
          
          

          // var t = this.widgets?.find((t) => t.name === 'tags')
          // if (t) {
          //     //console.log(t);
          //     t.element.innerText = `Item: ${item}, Index: ${index}`;
          //     this.onResize?.(this.size);  // onResize redraws the node
          // } else {
          //     const container = document.createElement("div");
          //     const text1 = document.createElement("span");
          //     text1.textContent = `Item: ${item}, Index: ${index}`;
          //     container.appendChild(text1);
  
          //     this.addDOMWidget("tags", "STRING", container, {
          //         //getMinHeight: () => 10,
          //         getHeight: () => 8
          //     });                
              
          //     this.size[1] += 20;
          //     this.onResize?.(this.size);
          // }  
            
            


        });

        return r;
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = async function (widget) {
        onConfigure?.apply(this, arguments);

        await this.getTitle();

        // console.log("🚀 Test1 node configured");
        // console.log(widget);

      };

    }
  },
});