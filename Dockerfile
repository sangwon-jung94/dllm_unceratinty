# -----------------------------------------------------
# Minimal, Persistent LLaDA Environment (RTX 6000, 8 GPU)
# Base: PyTorch 2.7.0 + CUDA 12.8 + cuDNN 9
# -----------------------------------------------------

FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# 기본 유틸 설치
RUN apt-get update && apt-get install -y \
    git wget vim curl build-essential procps tmux \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip setuptools wheel

# 주요 패키지 설치 (필요 최소 구성)
RUN pip install \
    transformers==4.45.2 \
    accelerate==1.1.1 \
    bitsandbytes==0.44.1 \
    peft==0.12.0 \
    datasets==3.1.0 \
    sentencepiece safetensors einops \
    tqdm wandb ninja matplotlib \
    && pip cache purge

# 사용자 편의 설정
RUN echo "export HF_HOME=/workspace/hf_cache" >> ~/.bashrc && \
    echo "alias zombie='ps -eo pid,ppid,state,cmd | grep defunct'" >> ~/.bashrc

# 작업 디렉토리
WORKDIR /workspace

# 최소한의 환경 변수
ENV TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

# init 프로세스 추가 (좀비 프로세스 방지)
# tini는 Docker 공식 권장 init (PID 1 문제 해결)
RUN apt-get update && apt-get install -y tini && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/usr/bin/tini", "--"]

# 기본 커맨드 (login shell)
CMD ["bash", "-li"]