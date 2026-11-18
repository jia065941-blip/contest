FROM python:3.11-slim-trixie

USER root

# 时区设置
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 先装依赖，利用缓存
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Linux 原生运行库。
COPY libUtils.so /usr/lib/
RUN ldconfig /usr/local/lib

# 复制项目代码
COPY . .

CMD ["python", "run_competition.py", "--config", "run_competition_config.yaml", "--blue_mode", "blue_defense/immediate/b0_fixed_ratio_random.py", "--scenario", "scenarios/Rtest2.json"]
