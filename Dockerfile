FROM python:3.10-slim

ENV TZ=Asia/Bangkok

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tzdata \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Fail the build immediately if pip's resolver pulled in a CUDA build of torch instead
# of the +cpu wheel pinned in requirements.txt (e.g. because rfdetr depends on
# torch>=2.2.0 without pinning a build) -- catches this at build time instead of via a
# disk-quota crash from ~4-5GB of unused nvidia-* CUDA packages on this CPU-only server.
RUN python -c "import torch; assert '+cpu' in torch.__version__, torch.__version__; assert not torch.cuda.is_available()"

# Also fail fast if rfdetr's own import chain is broken (e.g. its transitive
# `transformers` dependency silently disabling itself because it wants a newer torch
# than what's pinned above -- this happened once already: transformers 5.15.0 requires
# torch>=2.5, which the earlier torch==2.2.0+cpu pin didn't satisfy, and the failure
# only surfaced at container *run* time, not build time, wasting a full rebuild cycle).
RUN python -c "import rfdetr"

# Copy models directory first to take advantage of Docker layer caching
COPY models/ ./models/

COPY . .

EXPOSE 8932

CMD ["flask", "run", "--host=0.0.0.0", "--port=8932"] 