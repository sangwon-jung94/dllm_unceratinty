#!/bin/bash
# LLaDA 샘플링 순서 시각화 실행 예시

echo "========================================="
echo "LLaDA 샘플링 순서 시각화 도구"
echo "========================================="
echo ""
echo "사용 예시:"
echo ""
echo "1. 기본 실행 (대화형 프롬프트 선택):"
echo "   python visualize_sampling_order.py"
echo ""
echo "2. 특정 프롬프트로 실행:"
echo "   python visualize_sampling_order.py --prompt \"What is 2 + 2?\""
echo ""
echo "3. 생성 길이와 스텝 조정:"
echo "   python visualize_sampling_order.py --gen_length 64 --steps 64 --block_length 64"
echo ""
echo "4. 리마스킹 전략 변경:"
echo "   python visualize_sampling_order.py --remasking random"
echo ""
echo "========================================="
echo ""

read -p "예시 실행하시겠습니까? (y/n): " choice

if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    echo ""
    echo "간단한 예시 실행 중..."
    python visualize_sampling_order.py --gen_length 32 --steps 32 --block_length 32 --prompt "Hello, how are you?"
fi
