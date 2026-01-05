import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Sequence
import math

# 같은 폴더의 modeling_llada.py에서 필요한 요소들을 가져옵니다.
try:
    from . import modeling_llada
    from .modeling_llada import (
        LLaDALlamaBlock,
        LLaDAModel,
        LLaDAModelLM,
        LLaDAOutput,
        ActivationCheckpointingStrategy,
        get_causal_attention_bias,
        ensure_finite_,
        _non_meta_init_device,
        Dropout
    )
except ImportError:
    # Fallback for direct script execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import modeling_llada
    from modeling_llada import (
        LLaDALlamaBlock,
        LLaDAModel,
        LLaDAModelLM,
        LLaDAOutput,
        ActivationCheckpointingStrategy,
        get_causal_attention_bias,
        ensure_finite_,
        _non_meta_init_device,
        Dropout
    )

class EnsembleLLaDABlock(LLaDALlamaBlock):
    """
    마지막 레이어에서 MLP Dropout 앙상블을 지원하기 위해 확장된 LLaDALlamaBlock입니다.
    forward 메서드에 num_ensembles 인자가 추가되었습니다.
    Attention용 dropout과 MLP용 dropout을 분리했습니다.
    """
    def __init__(self, layer_id: int, config, cache, mlp_dropout_p: Optional[float] = None):
        super().__init__(layer_id, config, cache)
        # MLP 전용 dropout 추가 (attention dropout과 별도)
        # mlp_dropout_p가 지정되지 않으면 config.residual_dropout 사용
        dropout_p = mlp_dropout_p if mlp_dropout_p is not None else config.residual_dropout
        self.mlp_dropout = Dropout(dropout_p)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        num_ensembles: int = 1,
        memory_efficient: bool = False,
    ) -> Union[Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]], 
               Tuple[List[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        
        # 1. Attention Part (기존 로직 유지)
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        # Attention 결과 더하기 (Attention 전용 dropout 적용)
        x = x + self.dropout(att)

        # 2. FeedForward (MLP) Part
        og_x = x  # Residual을 위한 원본 저장
        
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)
        else:
            x = self.ff_norm(x)
            
        x, x_up = self.ff_proj(x), self.up_proj(x)
        
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)
        else:
            x = self.act(x)
            
        x = x * x_up
        x = self.ff_out(x) # MLP Output (Dropout 직전)

        # 3. Dropout Ensemble Logic
        # num_ensembles > 1일 때: 이미 계산된 MLP output(x)에 대해 dropout만 여러 번 적용
        # (Attention과 MLP forward는 위에서 한 번만 실행됨)
        if num_ensembles > 1:
            if memory_efficient:
                # Generator 방식: 메모리 효율적 (호출 시점에 하나씩 생성)
                def ensemble_generator():
                    for _ in range(num_ensembles):
                        # 매번 다른 MLP Dropout 마스크가 생성됩니다.
                        dropped_x = self.mlp_dropout(x)
                        yield og_x + dropped_x
                return ensemble_generator, cache, num_ensembles
            else:
                # 리스트 방식: 기존 호환성 유지 (모든 앙상블 결과를 메모리에 저장)
                ensemble_states = []
                for _ in range(num_ensembles):
                    dropped_x = self.mlp_dropout(x)
                    ensemble_states.append(og_x + dropped_x)
                return ensemble_states, cache
        else:
            # 일반적인 경우 (MLP 전용 dropout 사용)
            x = self.mlp_dropout(x)
            x = og_x + x
            return x, cache


class EnsembleLLaDAModel(LLaDAModel):
    """
    마지막 블록에 num_ensembles 인자를 전달하고, 앙상블 결과를 처리합니다.
    
    Args:
        memory_efficient: True이면 generator를 반환하여 메모리 효율적으로 처리.
                         False(기본값)이면 [num_ensembles, Batch, Seq, Vocab] 텐서 반환.
    """
    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        num_ensembles: int = 1,
        memory_efficient: bool = False,  # True: generator 반환, False: 텐서 반환
    ) -> LLaDAOutput:
        
        # --- [1. Init & Embedding Processing (원본 LLaDAModel 로직 복사)] ---
        assert not self.config.alibi, "Alibi length extrapolation is not supported for MDM."
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
        assert (past_key_values is None and not use_cache), "The kvcache is not suppotred for MDM."

        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings
        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)

        if not (self.config.alibi or self.config.rope):
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)
            x = pos_emb + x

        x = self.transformer.emb_drop(x)

        if attention_mask is not None and 0.0 in attention_mask:
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None

        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                attention_bias = self.get_bidirectional_attention_bias(past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        all_hidden_states = []
        
        # 앙상블 모드를 위한 변수
        ensemble_generator = None
        ensemble_count = 1

        # --- [2. Apply Blocks (수정된 부분)] ---
        # block_group_size == 1인 경우만 우선 지원합니다 (일반적인 구조)
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                
                # ** 마지막 레이어인지 확인 **
                is_last_layer = (block_idx == len(self.transformer.blocks) - 1)
                current_ensembles = num_ensembles if is_last_layer else 1
                
                # Checkpointing 적용 여부 확인
                use_checkpoint = (
                    (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                    or (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two and block_idx % 2 == 0)
                    or (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three and block_idx % 3 == 0)
                    or (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four and block_idx % 4 == 0)
                )

                if use_checkpoint:
                    # Checkpointing 사용 시 num_ensembles 인자를 넘기려면 partial 등을 이용해 래핑해야 하지만,
                    # 여기서는 단순화를 위해 Ensemble 모드일 때는 체크포인팅을 끄거나, 
                    # 혹은 forward 인자에 맞게 kwargs를 넘겨야 합니다.
                    # 여기서는 체크포인팅 로직 내부 수정 대신 직접 호출로 분기합니다.
                    # (앙상블 시 체크포인팅이 필요하다면 추가 구현 필요)
                    if current_ensembles > 1:
                        # 앙상블 블록은 직접 호출
                        result = block(
                            x, attention_bias=attention_bias, layer_past=layer_past, 
                            use_cache=use_cache, num_ensembles=current_ensembles,
                            memory_efficient=memory_efficient
                        )
                        if len(result) == 3:  # memory_efficient: generator, cache, count
                            ensemble_generator, cache, ensemble_count = result
                            x = None  # generator 모드에서는 x가 없음
                        else:  # 리스트 또는 일반 텐서
                            x, cache = result
                    else:
                        x, cache = self._activation_checkpoint_fn(
                            block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache
                        )
                else:
                    # 일반 호출 (여기서 num_ensembles 전달)
                    # 기존 블록일 경우 num_ensembles 인자를 받지 못해 에러가 날 수 있으므로 타입 체크
                    if isinstance(block, EnsembleLLaDABlock):
                        result = block(
                            x, attention_bias=attention_bias, layer_past=layer_past, 
                            use_cache=use_cache, num_ensembles=current_ensembles,
                            memory_efficient=memory_efficient
                        )
                        if len(result) == 3:  # memory_efficient: generator, cache, count
                            ensemble_generator, cache, ensemble_count = result
                            x = None  # generator 모드에서는 x가 없음
                        else:  # 리스트 또는 일반 텐서
                            x, cache = result
                    else:
                        x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
                
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
                
                # generator 모드면 루프 종료 (마지막 레이어)
                if ensemble_generator is not None:
                    break
        else:
            # Grouped blocks (여기서는 앙상블 미지원으로 남겨둠, 필요시 구현)
            raise NotImplementedError("Block group size > 1 not supported for ensemble mode yet.")

        # --- [3. Final Norm & Head (수정된 부분)] ---
        
        # 헬퍼 함수: Hidden State -> Logits 변환
        def compute_final_logits(hidden):
            if last_logits_only:
                hidden = hidden[:, -1, :].unsqueeze(1)
            
            hidden = self.transformer.ln_f(hidden)
            
            if output_hidden_states:
                all_hidden_states.append(hidden)
                
            if self.config.weight_tying:
                l = F.linear(hidden, self.transformer.wte.weight, None)
            else:
                l = self.transformer.ff_out(hidden)
                
            if self.config.scale_logits:
                l.mul_(1 / math.sqrt(self.config.d_model))
            return l

        if ensemble_generator is not None:
            # memory_efficient=True: Generator 반환
            # 메모리 효율성을 위해 모든 logits를 미리 계산하지 않고,
            # 필요할 때마다 하나씩 계산하여 반환하는 generator를 생성합니다.
            
            def logits_generator():
                for state in ensemble_generator():
                    # 각 앙상블 상태에 대해 독립적으로 Head 통과
                    yield compute_final_logits(state)
            
            # LLaDAOutput의 logits 필드에 generator를 할당
            # 호출하는 쪽(generate.py)에서 이를 인식하고 처리해야 함
            logits = logits_generator()
            
        elif isinstance(x, list):
            # memory_efficient=False: 리스트를 받아서 stacked tensor 반환
            # [num_ensembles, Batch, Seq, Vocab]
            logits_list = [compute_final_logits(state) for state in x]
            logits = torch.stack(logits_list, dim=0)
            
        else:
            # 일반 처리 (num_ensembles=1)
            logits = compute_final_logits(x)

        return LLaDAOutput(
            logits=logits, 
            attn_key_values=attn_key_values, 
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None
        )


def replace_last_block_with_ensemble(model: LLaDAModelLM, mlp_dropout_p: Optional[float] = None) -> LLaDAModelLM:
    """
    모델의 마지막 블록을 EnsembleLLaDABlock으로 교체합니다.
    기존 가중치를 복사하여 동일한 동작을 보장합니다.
    
    Args:
        model: LLaDAModelLM 모델
        mlp_dropout_p: 마지막 레이어 MLP dropout 확률 (None이면 config.residual_dropout 사용)
    """
    llada_model = model.model
    blocks = llada_model.transformer.blocks
    last_block = blocks[-1]
    
    # cache 접근 - LLaDABlock과 LLaDALlamaBlock 둘 다 __cache를 가지므로
    # name mangling된 속성을 찾아야 함
    cache = None
    for attr_name in ['_LLaDALlamaBlock__cache', '_LLaDABlock__cache']:
        if hasattr(last_block, attr_name):
            cache = getattr(last_block, attr_name)
            break
    
    if cache is None:
        raise RuntimeError("Could not find cache attribute in last block")
    
    # EnsembleLLaDABlock 생성 (같은 config와 cache 사용)
    ensemble_block = EnsembleLLaDABlock(
        layer_id=len(blocks) - 1,
        config=last_block.config,
        cache=cache,
        mlp_dropout_p=mlp_dropout_p
    )
    
    # 기존 블록의 가중치를 복사
    ensemble_block.load_state_dict(last_block.state_dict())
    
    # 디바이스와 dtype 맞추기
    ensemble_block = ensemble_block.to(
        device=next(last_block.parameters()).device,
        dtype=next(last_block.parameters()).dtype
    )
    
    # 마지막 블록 교체
    blocks[-1] = ensemble_block
    
    mlp_dropout_info = f" with mlp_dropout_p={mlp_dropout_p}" if mlp_dropout_p is not None else ""
    print(f"Successfully replaced last block (layer {len(blocks)-1}) with EnsembleLLaDABlock{mlp_dropout_info}")
    return model


def get_ensemble_model(model_path: str, mlp_dropout_p: Optional[float] = None, **kwargs):
    """
    모델을 로드한 후 마지막 블록을 EnsembleLLaDABlock으로 교체합니다.
    
    Monkey patching 대신 직접 블록 교체 방식을 사용합니다.
    이 방식이 더 안정적입니다 - from_pretrained가 호출될 때 이미 
    LLaDABlock.build()에서 LLaDALlamaBlock이 바인딩되어 있기 때문입니다.
    
    Args:
        model_path: 모델 경로
        mlp_dropout_p: 마지막 레이어 MLP dropout 확률 (None이면 config.residual_dropout 사용)
        **kwargs: from_pretrained에 전달될 추가 인자
    """
    print(f"Loading model from {model_path}...")
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)
    
    # 마지막 블록을 EnsembleLLaDABlock으로 교체
    model = replace_last_block_with_ensemble(model, mlp_dropout_p=mlp_dropout_p)
    
    # LLaDAModel의 forward를 EnsembleLLaDAModel의 forward로 교체
    print("Patching LLaDAModel.forward with EnsembleLLaDAModel.forward...")
    
    # EnsembleLLaDAModel의 forward 메서드를 바인딩
    import types
    model.model.forward = types.MethodType(EnsembleLLaDAModel.forward, model.model)
    
    return model

# 사용 예시 주석:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--test_only", action="store_true", help="간단한 구조 테스트만 수행")
    args = parser.parse_args()
    
    if args.test_only:
        # GPU 없이도 테스트 가능한 구조 검증
        print("=" * 50)
        print("EnsembleLLaDABlock 구조 테스트")
        print("=" * 50)
        
        # EnsembleLLaDABlock이 LLaDALlamaBlock을 상속하는지 확인
        print(f"EnsembleLLaDABlock bases: {EnsembleLLaDABlock.__bases__}")
        assert issubclass(EnsembleLLaDABlock, LLaDALlamaBlock), "EnsembleLLaDABlock should inherit from LLaDALlamaBlock"
        print("✓ EnsembleLLaDABlock correctly inherits from LLaDALlamaBlock")
        
        # EnsembleLLaDAModel이 LLaDAModel을 상속하는지 확인
        print(f"EnsembleLLaDAModel bases: {EnsembleLLaDAModel.__bases__}")
        assert issubclass(EnsembleLLaDAModel, LLaDAModel), "EnsembleLLaDAModel should inherit from LLaDAModel"
        print("✓ EnsembleLLaDAModel correctly inherits from LLaDAModel")
        
        # forward 메서드에 num_ensembles 인자가 있는지 확인
        import inspect
        sig = inspect.signature(EnsembleLLaDABlock.forward)
        assert "num_ensembles" in sig.parameters, "num_ensembles should be a parameter of EnsembleLLaDABlock.forward"
        print("✓ EnsembleLLaDABlock.forward has num_ensembles parameter")
        
        sig = inspect.signature(EnsembleLLaDAModel.forward)
        assert "num_ensembles" in sig.parameters, "num_ensembles should be a parameter of EnsembleLLaDAModel.forward"
        print("✓ EnsembleLLaDAModel.forward has num_ensembles parameter")
        
        print("=" * 50)
        print("모든 구조 테스트 통과!")
        print("=" * 50)
    else:
        # 전체 모델 로드 테스트 (GPU 필요)
        # MLP dropout 확률을 0.1로 설정
        model = get_ensemble_model(args.model_path, mlp_dropout_p=0.1, torch_dtype=torch.float16)
        model.cuda().eval()
        
        # 마지막 블록이 EnsembleLLaDABlock인지 확인
        last_block = model.model.transformer.blocks[-1]
        print(f"Last block type: {type(last_block)}")
        assert isinstance(last_block, EnsembleLLaDABlock), f"Expected EnsembleLLaDABlock, got {type(last_block)}"
        print("✓ Last block is EnsembleLLaDABlock")
        
        # MLP Dropout 확률 확인
        print(f"Last block MLP dropout p: {last_block.mlp_dropout.p}")
        assert abs(last_block.mlp_dropout.p - 0.1) < 1e-6, f"Expected mlp_dropout.p=0.1, got {last_block.mlp_dropout.p}"
        print("✓ MLP dropout probability is correctly set to 0.1")
        
        # 마지막 레이어 MLP Dropout 켜기
        last_block.mlp_dropout.train()

        input_ids = torch.randint(0, 100, (1, 10)).cuda()
        
        # Test 1: memory_efficient=False (default) - 리스트 모드
        print("\n[Test 1] memory_efficient=False (default)")
        output = model.model(input_ids, num_ensembles=5, memory_efficient=False)
        print(f"Output logits type: {type(output.logits)}")
        assert isinstance(output.logits, torch.Tensor), "Expected tensor"
        print(f"Output logits shape: {output.logits.shape}")
        assert output.logits.dim() == 4, f"Expected 4D tensor, got {output.logits.dim()}D"
        assert output.logits.shape[0] == 5, f"Expected 5 ensembles, got {output.logits.shape[0]}"
        print("✓ Stacked tensor mode works correctly!")
        
        # Test 2: memory_efficient=True - Generator 모드
        print("\n[Test 2] memory_efficient=True")
        output = model.model(input_ids, num_ensembles=5, memory_efficient=True)
        
        import types
        print(f"Output logits type: {type(output.logits)}")
        assert isinstance(output.logits, types.GeneratorType), "Expected generator"
        print("✓ Output is a generator!")
        
        # Generator 소비 테스트
        logits_list = list(output.logits)
        print(f"Generated {len(logits_list)} logits tensors")
        assert len(logits_list) == 5
        print(f"First logit shape: {logits_list[0].shape}")
        assert logits_list[0].dim() == 3  # [Batch, Seq, Vocab]
        print("✓ Generator mode works correctly!")
        
        print("\n" + "=" * 50)
        print("모든 테스트 통과!")
        print("=" * 50)