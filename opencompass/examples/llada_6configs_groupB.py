"""
6가지 remasking 설정 비교 - 그룹 B (HellaSwag + GSM8K + HumanEval + MBPP)

포함된 벤치마크 (~12,000 샘플):
- HellaSwag (10,042) - 상식
- GSM8K (1,319) - 수학
- HumanEval (164) - 코드
- MBPP (500) - 코드

사용법:
    cd /workspace/dllm_unceratinty/opencompass
    CUDA_VISIBLE_DEVICES=2,3 python run.py examples/llada_6configs_groupB.py
"""
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess
from opencompass.datasets import (
    HellaswagDataset_V2,
    GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator,
    HumanevalDataset, HumanEvalEvaluator, humaneval_postprocess_v2,
    MBPPDataset, MBPPEvaluator
)
from opencompass.models import LLaDAModel

# HellaSwag 설정
hellaswag_reader_cfg = dict(input_columns=['ctx', 'A', 'B', 'C', 'D'], output_column='label')
hellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='{ctx}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=16))
hellaswag_eval_cfg = dict(
    evaluator=dict(type=AccwithDetailsEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

hellaswag_datasets = [dict(
    abbr='hellaswag',
    type=HellaswagDataset_V2,
    path='opencompass/hellaswag',
    reader_cfg=hellaswag_reader_cfg,
    infer_cfg=hellaswag_infer_cfg,
    eval_cfg=hellaswag_eval_cfg)]

# GSM8K 설정
gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')
gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt="Question: {question}\nLet's think step by step\nAnswer:"),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))
gsm8k_eval_cfg = dict(
    evaluator=dict(type=Gsm8kEvaluator),
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_datasets = [dict(
    abbr='gsm8k',
    type=GSM8KDataset,
    path='opencompass/gsm8k',
    reader_cfg=gsm8k_reader_cfg,
    infer_cfg=gsm8k_infer_cfg,
    eval_cfg=gsm8k_eval_cfg)]

# HumanEval 설정
humaneval_reader_cfg = dict(input_columns=['prompt'], output_column='task_id')
humaneval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt='{prompt}')])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))
humaneval_eval_cfg = dict(
    evaluator=dict(type=HumanEvalEvaluator),
    pred_postprocessor=dict(type=humaneval_postprocess_v2))

humaneval_datasets = [dict(
    abbr='humaneval',
    type=HumanevalDataset,
    path='opencompass/humaneval',
    reader_cfg=humaneval_reader_cfg,
    infer_cfg=humaneval_infer_cfg,
    eval_cfg=humaneval_eval_cfg)]

# MBPP 설정
mbpp_reader_cfg = dict(input_columns=['text', 'test_list'], output_column='code')
mbpp_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='You are an expert Python programmer. Complete the following task:\n{text}\n\nTest cases:\n{test_list}\n\n```python\n'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))
mbpp_eval_cfg = dict(evaluator=dict(type=MBPPEvaluator))

mbpp_datasets = [dict(
    abbr='mbpp',
    type=MBPPDataset,
    path='opencompass/mbpp',
    reader_cfg=mbpp_reader_cfg,
    infer_cfg=mbpp_infer_cfg,
    eval_cfg=mbpp_eval_cfg)]

datasets = hellaswag_datasets + gsm8k_datasets + humaneval_datasets + mbpp_datasets

MODEL_PATH = 'GSAI-ML/LLaDA-8B-Instruct'
NUM_GPUS = 1

# Data parallel: 각 worker가 서로 다른 GPU에서 독립적으로 실행
# device_map='cuda:0' - OpenCompass가 각 worker에 CUDA_VISIBLE_DEVICES=N 설정하므로
# cuda:0이 해당 worker에 할당된 GPU를 가리킴
common_config = dict(
    type=LLaDAModel,
    path=MODEL_PATH,
    max_out_len=256,
    batch_size=1,
    run_cfg=dict(num_gpus=NUM_GPUS),
    model_kwargs=dict(device_map='cuda:0', torch_dtype='torch.bfloat16'),
    gen_steps=256,
    gen_length=256,
    gen_blocksize=256,
    diff_confidence_eos_eot_inf=True,
    diff_logits_eos_inf=True,
    memory_efficient=True,
)

models = [
    dict(**common_config, abbr='llada-8b-low_confidence', remasking='low_confidence', use_ensemble_llada=False),
    dict(**common_config, abbr='llada-8b-entropy', remasking='entropy', use_ensemble_llada=False),
    dict(**common_config, abbr='llada-8b-ua-a1.0-b1.0', remasking='uncertainty_aware', use_ensemble_llada=True, mc_samples=8, alpha=1.0, beta=1.0, mlp_dropout_p=0.1),
    dict(**common_config, abbr='llada-8b-ua-a0.0-b1.0', remasking='uncertainty_aware', use_ensemble_llada=True, mc_samples=8, alpha=0.0, beta=1.0, mlp_dropout_p=0.1),
    dict(**common_config, abbr='llada-8b-ua-a1.0-b0.0', remasking='uncertainty_aware', use_ensemble_llada=True, mc_samples=8, alpha=1.0, beta=0.0, mlp_dropout_p=0.1),
    dict(**common_config, abbr='llada-8b-ua-a1.0-b0.0-clean', remasking='uncertainty_aware', use_ensemble_llada=True, mc_samples=8, alpha=1.0, beta=0.0, mlp_dropout_p=0.1, sample_from_clean_logits=True),
]

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

# num_worker=2: 2개 GPU 사용 (CUDA_VISIBLE_DEVICES=2,3이면 각각 GPU 2, 3 사용)
# max_num_workers=2: 동시에 최대 2개 작업 실행
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=4, num_split=None, min_task_size=16),
    runner=dict(type=LocalRunner, max_num_workers=4, task=dict(type=OpenICLInferTask), retry=5),
)
