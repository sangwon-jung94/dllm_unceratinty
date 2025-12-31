#!/bin/bash
# 모든 GSM8K 결과를 채점하는 스크립트

# 첫 번째 인자로 max_samples 받기
MAX_SAMPLES="$1"

if [ -n "$MAX_SAMPLES" ]; then
    echo "Evaluating GSM8K results (max $MAX_SAMPLES samples per experiment)..."
else
    echo "Evaluating all GSM8K results..."
fi
echo "================================"

# result 디렉토리의 모든 gsm8k 관련 폴더 찾기
for result_dir in ./result/gsm8k_*/; do
    if [ -f "${result_dir}output.txt" ]; then
        exp_name=$(basename "$result_dir")
        echo ""
        echo "Evaluating: $exp_name"
        echo "-------------------"
        
        # 채점 실행 및 결과 저장
        if [ -n "$MAX_SAMPLES" ]; then
            python3 evaluate_gsm8k.py \
                --output_file "${result_dir}output.txt" \
                --save_results "${result_dir}evaluation.json" \
                --max_samples "$MAX_SAMPLES"
        else
            python3 evaluate_gsm8k.py \
                --output_file "${result_dir}output.txt" \
                --save_results "${result_dir}evaluation.json"
        fi
        
        echo ""
    fi
done

echo ""
echo "================================"
echo "All evaluations completed!"
echo ""
echo "Summary:"
echo "--------"
# 모든 결과의 accuracy를 한번에 보기
for result_dir in ./result/gsm8k_*/; do
    if [ -f "${result_dir}evaluation.json" ]; then
        exp_name=$(basename "$result_dir")
        accuracy=$(python3 -c "import json; f=open('${result_dir}evaluation.json'); d=json.load(f); print(f\"{d['accuracy']:.2f}%\")")
        echo "$exp_name: $accuracy"
    fi
done
