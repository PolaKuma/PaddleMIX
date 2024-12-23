# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import paddle

from paddlenlp.transformers import JambaConfig
from paddlenlp.transformers.jamba.modeling import JambaModel, JambaForCausalLM, JambaLMHead
from paddlenlp.transformers.model_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

__all__ = [
    "LlavaJambaConfig",
    "LlavaJambaModel",
    "LlavaJambaForCausalLM",
]


class LlavaJambaConfig(JambaConfig):
    model_type = "llava_jamba"


class LlavaJambaModel(LlavaMetaModel, JambaModel):
    config_class = LlavaJambaConfig

    def __init__(self, config: JambaConfig):
        super(LlavaJambaModel, self).__init__(config)


class LlavaJambaForCausalLM(JambaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaJambaConfig
    base_model_prefix = "llava"

    def __init__(self, config):
        super(JambaForCausalLM, self).__init__(config)
        self.jamba = LlavaJambaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = JambaLMHead(config)

    def get_model(self):
        return self.jamba

    def forward(
            self,
            input_ids: paddle.Tensor = None,
            attention_mask: Optional[paddle.Tensor] = None,
            position_ids: Optional[paddle.Tensor] = None,
            past_key_values: Optional[List[paddle.Tensor]] = None,
            inputs_embeds: Optional[paddle.Tensor] = None,
            labels: Optional[paddle.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[paddle.Tensor] = None,
            image_sizes: Optional[List[List[int]]] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    @paddle.no_grad()
    def generate(
            self,
            inputs: Optional[paddle.Tensor] = None,
            images: Optional[paddle.Tensor] = None,
            image_sizes: Optional[paddle.Tensor] = None,
            **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        modelDtype = self.get_model().dtype
        inputs_embeds = inputs_embeds.to(dtype=modelDtype)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
