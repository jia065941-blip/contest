# docker build -t competition:v1 .

FROM python3.11-app:latest

USER root

# 安装 tzdata 并设置时区
RUN apt update && apt install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV TZ=Asia/Shanghai

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

COPY ./libUtils.so /usr/lib/

# 复制项目代码
COPY . .

# 设置入口点
CMD ["python3", "main.py"]