# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Public API for retrieving registered architectures from the Hugging Face Hub
if implemented in this library.
"""

from __future__ import annotations

from ureinuz.maestro import opus
from ureinuz.maestro._livret import repertoire

from jax.sharding import Mesh
from jax.experimental import mesh_utils
from huggingface_hub import hf_hub_download
import json


class Maestro:
    @classmethod
    def from_pretrained(
        cls, 
        repo_or_path, 
        mesh=None, 
        sharding_rules=None, 
        local=False,
        **kwargs
    ):
        try:
            config_path = hf_hub_download(repo_or_path, 'config.json')
            with open(config_path, 'r') as config_file:
                config_dict = json.load(config_file)

        except Exception as e:
            print(f'{e}')
            exit(0)

        keys = config_dict.get('architectures', [])
        
        assert len(keys) == 1, \
            'Unsupported architectures.'

        key = keys[0]
        if key not in repertoire.available():
            raise NotImplementedError("Unsupported architectures.")
            
        model_cls = repertoire.get(key)
        
        # Parse Mesh if provided as a dict (e.g. {'data': 4, 'model': 2})
        if isinstance(mesh, dict):
            axis_names = tuple(mesh.keys())
            shape = tuple(mesh.values())
            devices = mesh_utils.create_device_mesh(shape)
            mesh = Mesh(devices, axis_names)
            
        if sharding_rules is None and hasattr(model_cls, 'default_sharding_rules'):
            sharding_rules = model_cls.default_sharding_rules
            
        return model_cls.from_pretrained(
            repo_or_path, 
            mesh=mesh, 
            sharding_rules=sharding_rules, 
            local=local,
            **kwargs
        )


__all__ = ['Maestro']