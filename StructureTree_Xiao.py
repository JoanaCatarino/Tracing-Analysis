import json
import numpy as np
from multipledispatch import dispatch

class StructureTree():
    def __init__(self,
                 structures_json_path="C:/Users/xiao/.brainglobe/allen_mouse_10um_v1.2/structures.json") -> None:
        class Acronym(list):
            def __init__(self,*args,**kwargs) -> None:
                self.stree = None
                return super().__init__(self,*args,**kwargs)
            
            @dispatch((list,np.ndarray))
            def from_id(self,id) -> np.ndarray:
                map_dict = dict(zip(self.stree.id,self)) # "id" needs to be unique
                return np.vectorize(map_dict.__getitem__)(id)
            
            @dispatch((int,np.int64,np.int32))
            def from_id(self,id) -> int:
                map_dict = dict(zip(self.stree.id,self)) # "id" needs to be unique
                return map_dict[id]
            
            @dispatch((list,np.ndarray))
            def startswith(self,acronym) -> list:
                query = []
                for acr in acronym:
                    query_temp = []
                    for ac in self:
                        if ac.startswith(acr):
                            query_temp.append(ac)
                    query.append(query_temp)
                return query
            
            @dispatch(str)
            def startswith(self,acronym) -> list:
                query = []
                for ac in self:
                    if ac.startswith(acronym):
                        query.append(ac)
                return query
                    

            

            


        class ID(list):
            def __init__(self,*args,**kwargs) -> None:
                self.stree = None
                return super().__init__(self,*args,**kwargs)
            
            @dispatch((list,np.ndarray))
            def from_acronym(self,acronym) -> list:
                map_dict = dict(zip(self.stree.acronym,self))
                return np.vectorize(map_dict.__getitem__)(acronym)
            
            @dispatch(str)
            def from_acronym(self,acronym) -> int:
                map_dict = dict(zip(self.stree.acronym,self))
                return map_dict[acronym]


            
        class Name(list):
            def __init__(self,*args,**kwargs) -> None:
                return super().__init__(self,*args,**kwargs)
            
        class StructureIdPath(list):
            def __init__(self,*args,**kwargs) -> None:
                return super().__init__(self,*args,**kwargs)
            
        class RGBTriplet(list):
            def __init__(self,*args,**kwargs) -> None:
                self.stree = None
                return super().__init__(self,*args,**kwargs)
            
            @dispatch((int,np.uintc))
            def from_id(self,id) -> list:
                map_dict = dict(zip(self.stree.id,self.stree.rgb_triplet))
                return map_dict[id]
            
            @dispatch((list,np.ndarray))
            def from_id(self,id) -> np.ndarray:
                map_dict = dict(zip(self.stree.id,self.stree.rgb_triplet))
                return np.array(list(map(map_dict.__getitem__,id)),dtype=np.uint8)
        

        with open(structures_json_path,"r") as f:
            self.df_tree = json.load(f)
            # add id:0 is root rule
            self.df_tree.append({"acronym": "Root", 
                                 "id": 0, 
                                 "name": "Root", 
                                 "structure_id_path": [0], 
                                 "rgb_triplet": [255, 0, 0]})
            
            self.acronym = Acronym()
            self.id = ID()
            self.name = Name()
            self.structure_id_path = StructureIdPath()
            self.rgb_triplet = RGBTriplet()

            self.acronym.stree = self
            self.id.stree = self
            self.name.stree = self
            self.structure_id_path.stree = self
            self.rgb_triplet.stree = self

            for record in self.df_tree:
                self.acronym.append(record["acronym"])
                self.id.append(record["id"])
                self.name.append(record["name"])
                self.structure_id_path.append(record["structure_id_path"])
                self.rgb_triplet.append(record["rgb_triplet"])
            



    

