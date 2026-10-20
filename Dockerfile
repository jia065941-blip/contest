# 建议改用官方镜像，或确保自定义基础镜像已推送到仓库
FROM python:3.11-slim

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
RUN pip3 install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 复制 so 库（确认架构匹配！）
COPY core/libUtils.so /usr/lib/
RUN ldconfig /usr/local/lib

# 复制项目代码
COPY . .

CMD ["python", "run.py", "run", "--scenario", "easy/E01", "--red-policy", "r8_satellite_packages", "--blue-policy", "b3_min_cost_assignment", "--red-motion-policy", "reactive_evasion", "--run-id", "demo", "--", "--total-rounds", "1", "--render-mode", "none", "--disable-log-color", "--enable-config", "--enable-state", "--enable-event"]