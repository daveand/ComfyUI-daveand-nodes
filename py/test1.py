from server import PromptServer
from aiohttp import web
routes = PromptServer.instance.routes
import hashlib
import time

items = ["item1", "item2", "item3"]

class Test1:
    CATEGORY = "utils"
    @classmethod    
    def INPUT_TYPES(s):
       
        return { 
            "required":  {
                "item": (items, ),
                "index": ("INT", ),
            }
        }
    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("index", )
    FUNCTION = "test_1"
    OUTPUT_NODE = True

    # @classmethod
    # def IS_CHANGED(self, item, index):
    #     m = hashlib.sha256()

    #     m.update(item)
    #     m.update(index)

    #     return m.digest().hex()

    def test_1(self, item, index):

        index_selected = items.index(item)

        index = index_selected

        PromptServer.instance.send_sync(
            "daveand_data_to_ui", {"item": item, "index": index}
        )

        #PromptServer.instance.send_sync("daveand.test1.textmessage", {"message":f"Hello from backend! Selected index: {item}"})

        return (index, )
