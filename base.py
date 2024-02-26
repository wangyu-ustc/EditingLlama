import torch

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import logging
from abc import ABC

logger = logging.get_logger(__name__)

class MemoryLMOutputWithPastAndCrossAttentions(CausalLMOutputWithCrossAttentions):
    def __init__(
        self,
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
        cross_attentions=None,
        delta_memory=None,
        last_hidden_state=None,
        remaining_indices=None
    ):
        super().__init__(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            cross_attentions=cross_attentions,
        )
        self.delta_memory = delta_memory
        self.remaining_indices = remaining_indices
        self.last_hidden_state = last_hidden_state


class BaseMemoryModel(ABC):
    def __init__(self, config):
        self.config = config

    def inject_memory(self, context_ids, 
                            context_attention_masks,
                            delta_memory=None,
                            update_memory=False):

        output = self(input_ids=context_ids,
                attention_mask=context_attention_masks,
                delta_memory=delta_memory,
                output_delta_memory=True,
                return_dict=True)
        
        if update_memory:
            self.update_memory_with_delta_memory(output.delta_memory, remaining_indices=output.remaining_indices)

            return output.delta_memory

        else:
            return output.delta_memory
    
    def cat_memory_and_hiddens(self, i,
                               hidden_states,
                               delta_memory=None,
                               is_injection=True,
                               remaining_indices=None,
                               gradient_descent_block_idx=None):


        if not self.initialized:
            return hidden_states

        if self.initialized:

            if delta_memory is None or len(delta_memory) == 0:
                # It means we are in the first step of injection
                cur_memory = self.memory[i].unsqueeze(0).repeat(len(hidden_states), 1, 1)
                if cur_memory.device != hidden_states:
                    cur_memory = cur_memory.to(hidden_states.device)
                
            else:
                cur_memory = delta_memory[:, i]

                if self.delta_memory_ratio < 1:
                    old_memory = self.memory[i].unsqueeze(0).repeat(len(hidden_states), 1, 1) * (1 - self.delta_memory_ratio)
                    if old_memory.device != hidden_states:
                        old_memory = old_memory.to(hidden_states.device)
                    cur_memory = cur_memory * self.delta_memory_ratio + old_memory * (1 - self.delta_memory_ratio)
                
                if remaining_indices is not None:
                    cur_memory = torch.cat([
                        self.memory[i][remaining_indices].unsqueeze(0).repeat(len(hidden_states), 1, 1),
                        cur_memory
                    ], dim=1)

        if self.add_positional_embedding:
            if self.split_positional_embedding:
                if is_injection:
                    positional_embeddings = self.memory_positional_emb[0](self.positional_indices.to(self.memory.device))
                else:
                    positional_embeddings = self.memory_positional_emb[1](self.positional_indices.to(self.memory.device))
                
            else:
                positional_embeddings = self.memory_positional_emb(self.positional_indices.to(self.memory.device))

            if gradient_descent_block_idx is not None:
                    
                positional_embeddings[:, :gradient_descent_block_idx*self.num_tokens] = \
                    positional_embeddings[:, :gradient_descent_block_idx*self.num_tokens].detach()
                positional_embeddings[:, (gradient_descent_block_idx+1)*self.num_tokens:] = \
                    positional_embeddings[:, (gradient_descent_block_idx+1)*self.num_tokens:].detach()

            cur_memory = cur_memory + positional_embeddings

        if self.add_bos_embedding:
            cur_memory = torch.cat([self.bos_embedding[i].unsqueeze(0).repeat(len(cur_memory), 1, 1), cur_memory], dim=1)

        return torch.cat([cur_memory, hidden_states], dim=1)


    def update_memory_with_delta_memory(self, delta_memory, remaining_indices=None):

        if self.add_pad_token:

            self.memory.data = delta_memory[0]

            # assert len(delta_memory[0].shape) == 3
            # assert delta_memory[0].shape[1] == self.num_tokens
            # delta_memory = [x.detach().to(self.memory.device) for x in delta_memory]
            # self.memory.data = torch.cat(delta_memory, dim=0)
            
            if not self.initialized:
                self.initialized += 1

        else:
            
            if remaining_indices is None or self.initialized == 0:
                assert len(delta_memory[0].shape) == 3
                delta_memory = delta_memory.detach()[0]
                # delta_memory = [x.detach().to(self.memory.device) for x in delta_memory]
                # delta_memory = torch.cat(delta_memory, dim=0)

                if delta_memory.shape[1] < self.num_tokens:
                    if (self.num_tokens % delta_memory.shape[1]) == 0:
                        delta_memory = torch.cat(
                            [delta_memory] * (self.num_tokens // delta_memory.shape[1]), dim=1
                        )
                    else:
                        delta_memory = torch.cat(
                            [delta_memory] * (self.num_tokens // delta_memory.shape[1]) + 
                            [delta_memory[:, -(self.num_tokens % delta_memory.shape[1]):]], dim=1
                        )
                
                if not self.memory.device == 'cpu':
                    if self.delta_memory_ratio == 1:
                        self.memory.data = delta_memory
                    else:
                        self.memory.data = delta_memory * self.delta_memory_ratio + self.memory.data * (1 - self.delta_memory_ratio)
                else:
                    if self.dalta_memory_ratio == 1:
                        self.memory = delta_memory
                    else:
                        self.memory = delta_memory * self.delta_memory_ratio + self.memory * (1 - self.delta_memory_ratio)
                
            else:
                assert len(delta_memory[0].shape) == 3
                delta_memory = [x.detach().to(self.memory.device) for x in delta_memory]
                delta_memory = torch.cat(delta_memory, dim=0)
                if self.memory.device == 'cpu':
                    self.memory = torch.cat([
                        self.memory[:, remaining_indices], 
                        delta_memory
                    ], dim=1)
                else:
                    self.memory.data = torch.cat([
                        self.memory.data[:, remaining_indices], 
                        delta_memory
                    ], dim=1)

            if not self.initialized:
                self.initialized += 1

    def customized_generate(
        self, 
        inputs_ids,
        inputs_masks,
        tokenizer,
        max_new_tokens,
        delta_memory=None,
    ):

        assert len(inputs_ids) == 1, "We currently only support generation with batch_size=1"

        count = 0
        while (not inputs_ids[0][-1].eq(tokenizer.eos_token_id)) and count < max_new_tokens:

            model_outputs = self(
                input_ids=inputs_ids,
                attention_mask=inputs_masks,
                delta_memory=delta_memory,
                output_delta_memory=False,
                return_dict=True
            )

            count += 1

            new_id = model_outputs.logits[0][-1].argmax()

            inputs_ids = torch.cat(
                [inputs_ids,
                torch.tensor([new_id]).unsqueeze(0).to(inputs_ids.device)], dim=-1
            )
            inputs_masks = torch.cat(
                [inputs_masks,
                torch.tensor([1]).unsqueeze(0).to(inputs_masks.device)], dim=-1
            )

        return inputs_ids
