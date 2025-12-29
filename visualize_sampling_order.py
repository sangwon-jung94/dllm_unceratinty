"""
LLaDA 모델의 샘플링 순서 시각화 스크립트

각 스텝별로 생성되는 시퀀스를 추적합니다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import os


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


def enable_mc_dropout(model, p=None):
    """MC Dropout을 위해 dropout 레이어 활성화"""
    dropout_count = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Dropout) and not name.endswith("emb_drop"):
            if p is not None:
                m.p = p
            m.train()
            dropout_count += 1
        # if m.__class__.__name__ == "LLaDALlamaBlock":
        #     print(f"Enabling MC Dropout in {name}, setting attention_dropout to {p if p is not None else m.config.attention_dropout}")
        #     m.config.attention_dropout = p if p is not None else m.config.attention_dropout
        #     m.train()
        #     dropout_count += 1
    return dropout_count


def disable_mc_dropout(model):
    """Dropout 레이어 비활성화"""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
        if m.__class__.__name__ == "LLaDALlamaBlock":
            m.eval()
            m.config.attention_dropout = 0.
    


def compute_entropy(probs, dim=-1, eps=1e-9):
    """엔트로피 계산: H(p) = -sum(p * log(p))"""
    return (-probs * probs.clamp_min(eps).log()).sum(dim=dim)


def compute_uncertainty_decomposition(model, x, attention_mask, mc_samples, cfg_scale, prompt_index, mask_id, dropout_p=None):
    """MC Dropout을 통한 epistemic/aleatoric 불확실성 분해"""
    probs_sum = None
    entropy_sum = None
    logits_sum = None

    dropout_count = enable_mc_dropout(model, p=dropout_p)

    try:
        for k in range(mc_samples):
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    logits = model(x_, attention_mask=attention_mask_).logits
                else:
                    logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                if attention_mask is not None:
                    logits = model(x, attention_mask=attention_mask).logits
                else:
                    logits = model(x).logits

            p = F.softmax(logits, dim=-1)
            h_k = compute_entropy(p, dim=-1)

            if probs_sum is None:
                probs_sum = p.clone()
                entropy_sum = h_k.clone()
                logits_sum = logits.clone()
            else:
                probs_sum = probs_sum + p
                entropy_sum = entropy_sum + h_k
                logits_sum = logits_sum + logits

    finally:
        disable_mc_dropout(model)

    p_bar = probs_sum / mc_samples
    mean_logits = logits_sum / mc_samples
    H_total = compute_entropy(p_bar, dim=-1)
    H_aleatoric = entropy_sum / mc_samples
    H_epistemic = (H_total - H_aleatoric).clamp_min(0.0)

    return mean_logits, H_epistemic, H_aleatoric, p_bar


def compute_uncertainty_score(epistemic, aleatoric, alpha, beta):
    """불확실성 점수 계산: 높은 점수 = 낮은 불확실성 = 먼저 언마스킹"""
    return -alpha * epistemic - beta * aleatoric


@torch.no_grad()
def generate_with_tracking(model, prompt, tokenizer, attention_mask=None, steps=128, gen_length=128, 
                          block_length=128, temperature=0., cfg_scale=0., remasking='low_confidence', 
                          mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False,
                          mc_samples=8, alpha=1.0, beta=1.0, dropout_p=None):
    '''
    각 스텝의 시퀀스 상태를 추적하면서 생성
    
    Args:
        mc_samples: MC Dropout 샘플 수 (uncertainty_aware 모드에서만 사용)
        alpha: Epistemic 불확실성 가중치
        beta: Aleatoric 불확실성 가중치
        dropout_p: Dropout 확률 (None이면 기존 레이어 확률 사용)
        logits_eos_inf: EOS 토큰 logits를 -inf로 설정하여 조기 종료 방지
        confidence_eos_eot_inf: EOS/EoT 토큰의 confidence를 -inf로 설정
    '''
    # Dropout 체크
    if remasking == 'uncertainty_aware':
        dropout_count = sum(1 for m in model.modules() if isinstance(m, nn.Dropout))
        if dropout_count == 0:
            warnings.warn(
                "No dropout layers found. Falling back to 'low_confidence' remasking.",
                UserWarning
            )
            remasking = 'low_confidence'
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    # 각 스텝에서의 시퀀스 상태 저장
    x_history = [x[0].clone().cpu()]
    
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), 
                                                                dtype=attention_mask.dtype, 
                                                                device=model.device)], dim=-1)
    
    prompt_index = (x != mask_id)
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            
            # 모델 추론 및 Remasking 전략
            if remasking == 'uncertainty_aware':
                # MC Dropout으로 불확실성 분해
                mean_logits, H_epistemic, H_aleatoric, p_bar = compute_uncertainty_decomposition(
                    model=model,
                    x=x,
                    attention_mask=attention_mask,
                    mc_samples=mc_samples,
                    cfg_scale=cfg_scale,
                    prompt_index=prompt_index,
                    mask_id=mask_id,
                    dropout_p=dropout_p
                )
                logits = mean_logits
                
                if logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                
                if confidence_eos_eot_inf:
                    logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf
                
                # 불확실성 기반 점수 계산
                x0_p = compute_uncertainty_score(H_epistemic, H_aleatoric, alpha, beta)
            else:
                # 기존 방식 (low_confidence, random)
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    if attention_mask is not None:
                        attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                        logits = model(x_, attention_mask=attention_mask_).logits
                    else:
                        logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    if attention_mask is not None:
                        logits = model(x, attention_mask=attention_mask).logits
                    else:
                        logits = model(x).logits
                
                if logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                
                if confidence_eos_eot_inf:
                    logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf
                
                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)
            
            x0_p[:, block_end:] = -np.inf
            
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # 언마스킹할 토큰 선택
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            
            # 토큰 업데이트
            x[transfer_index] = x0[transfer_index]
            
            # 현재 스텝의 시퀀스 상태 저장
            x_history.append(x[0].clone().cpu())
    
    return x, x_history


def save_sequence_by_step(prompt, generated, x_history, tokenizer, save_dir):
    """각 스텝별로 생성되는 시퀀스를 저장"""
    os.makedirs(save_dir, exist_ok=True)
    
    # EOS/EOT 토큰 ID들
    EOS_TOKENS = {126081, 126348}  # <|endoftext|>, <|eot_id|>
    MASK_ID = 126336
    
    prompt_len = prompt.shape[1]
    
    analysis_file = os.path.join(save_dir, 'sequence_by_step.txt')
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("LLaDA 샘플링 과정 - 각 스텝별 시퀀스 상태\n")
        f.write("=" * 100 + "\n\n")
        
        # 프롬프트 출력
        f.write("■ 프롬프트:\n")
        prompt_text = tokenizer.decode(prompt[0], skip_special_tokens=True)
        f.write(f"{prompt_text}\n\n")
        
        # 최종 생성 텍스트 출력
        f.write("■ 최종 생성 텍스트:\n")
        final_tokens = [t for t in generated[0, prompt_len:].cpu().tolist() if t not in EOS_TOKENS]
        if final_tokens:
            generated_text = tokenizer.decode(final_tokens, skip_special_tokens=True)
            f.write(f"{generated_text}\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("■ 각 스텝별 생성 시퀀스\n")
        f.write("=" * 100 + "\n\n")
        
        # 각 스텝의 시퀀스 상태 출력
        for step_idx, x_state in enumerate(x_history):
            # 생성 부분만 추출
            gen_tokens = x_state[prompt_len:].tolist()
            
            # EOS 토큰 제외하고 실제 생성된 텍스트만 추출
            actual_tokens = [t for t in gen_tokens if t not in EOS_TOKENS and t != MASK_ID]
            
            if actual_tokens:
                sequence_text = tokenizer.decode(actual_tokens, skip_special_tokens=True)
                if sequence_text.strip():
                    f.write(f"Step {step_idx}: {sequence_text}\n")
            else:
                if step_idx == 0:
                    f.write(f"Step {step_idx}: [모두 MASK]\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("■ 각 스텝별 토큰 상태 (상세)\n")
        f.write("=" * 100 + "\n\n")
        
        # 상세 토큰 정보
        for step_idx, x_state in enumerate(x_history[::max(1, len(x_history)//20)]):  # 20개 샘플만
            gen_tokens = x_state[prompt_len:].tolist()
            
            f.write(f"\n--- Step {step_idx * max(1, len(x_history)//20)} ---\n")
            
            tokens_display = []
            for i, t in enumerate(gen_tokens):
                if t == MASK_ID:
                    tokens_display.append("[MASK]")
                elif t not in EOS_TOKENS:
                    tok_str = tokenizer.decode([t])
                    tokens_display.append(f"{repr(tok_str)[1:-1]}")
            
            # 한 줄에 10개씩
            for i in range(0, len(tokens_display), 10):
                f.write(" ".join(tokens_display[i:i+10]) + "\n")
    
    print(f"\n분석 결과가 저장되었습니다: {analysis_file}")
    return analysis_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LLaDA 샘플링 과정 시각화')
    parser.add_argument('--prompt', type=str, default=None, help='프롬프트')
    parser.add_argument('--model', type=str, default='GSAI-ML/LLaDA-8B-Instruct', help='모델 경로')
    parser.add_argument('--device', type=str, default='cuda:7', help='디바이스')
    parser.add_argument('--gen_length', type=int, default=64, help='생성 길이')
    parser.add_argument('--steps', type=int, default=64, help='샘플링 스텝')
    parser.add_argument('--block_length', type=int, default=64, help='블록 길이')
    parser.add_argument('--temperature', type=float, default=0., help='온도')
    parser.add_argument('--cfg_scale', type=float, default=0., help='CFG 스케일')
    parser.add_argument('--remasking', type=str, default='low_confidence', 
                       choices=['low_confidence', 'random', 'uncertainty_aware'], help='리마스킹 전략')
    parser.add_argument('--mc_samples', type=int, default=8, help='MC Dropout 샘플 수 (uncertainty_aware용)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Epistemic 불확실성 가중치')
    parser.add_argument('--beta', type=float, default=1.0, help='Aleatoric 불확실성 가중치')
    parser.add_argument('--dropout_p', type=float, default=None, help='Dropout 확률 (None=기본값 사용)')
    parser.add_argument('--logits_eos_inf', action='store_true', 
                       help='EOS 토큰 logits를 -inf로 설정하여 조기 종료 방지')
    parser.add_argument('--confidence_eos_eot_inf', action='store_true',
                       help='EOS/EoT 토큰의 confidence를 -inf로 설정')
    parser.add_argument('--cache_dir', type=str, default=None, help='HF 캐시 경로(기본: /tmp/hf_cache)')
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("LLaDA 샘플링 과정 시각화")
    print("=" * 100)
    print(f"모델: {args.model}")
    print(f"생성 길이: {args.gen_length}, 스텝: {args.steps}, 블록: {args.block_length}")
    print(f"리마스킹: {args.remasking}")
    print("=" * 100)
    
    # 캐시/출력 경로 설정 (NAS I/O 회피)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_dir = args.cache_dir if args.cache_dir else os.environ.get('TRANSFORMERS_CACHE', '/workspace/hf_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # 모델 로딩
    print("\n모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, 
                                     torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(args.device).eval()
    print("로딩 완료!")
    
    # 프롬프트
    if args.prompt is None:
        prompts = [
            "What is 2 + 2?",
            "The capital of France is",
            "Explain quantum computing:",
        ]
        print("\n예시 프롬프트:")
        for i, p in enumerate(prompts, 1):
            print(f"{i}. {p}")
        choice = input("선택 (1-3) 또는 직접 입력: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(prompts):
            user_prompt = prompts[int(choice) - 1]
        else:
            user_prompt = choice if choice else prompts[0]
    else:
        user_prompt = args.prompt
    
    print(f"\n프롬프트: {user_prompt}")
    
    # 포맷팅
    messages = [{"role": "user", "content": user_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(args.device)
    
    print(f"\n생성 시작... (프롬프트 길이: {input_ids.shape[1]} 토큰)")
    
    # 생성
    generated, x_history = generate_with_tracking(
        model=model,
        prompt=input_ids,
        tokenizer=tokenizer,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        logits_eos_inf=args.logits_eos_inf,
        confidence_eos_eot_inf=args.confidence_eos_eot_inf,
        mc_samples=args.mc_samples,
        alpha=args.alpha,
        beta=args.beta,
        dropout_p=args.dropout_p
    )
    
    print("생성 완료!")
    
    # 생성 텍스트 출력
    generated_text = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n생성된 텍스트:\n{generated_text}")
    
    # 결과 저장 (기본적으로 로컬 tmp로 저장하여 NAS I/O 최소화)
    save_dir = f"sampling_analysis/sampling_analysis_{timestamp}"
    
    print("\n결과 저장 중...")
    save_sequence_by_step(input_ids, generated, x_history, tokenizer, save_dir)
    
    print("\n완료!")


if __name__ == "__main__":
    main()
