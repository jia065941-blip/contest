FROM ubuntu:22.04

ENV PYTHON_VERSION=3.11.15
ENV DEBIAN_FRONTEND=noninteractive

# 配置 apt 国内源
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list \
    && sed -i 's@//.*security.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list

# 安装依赖
RUN apt update && apt install -y \
    wget \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 下载 Python（使用华为云镜像）
WORKDIR /tmp
RUN wget https://mirrors.huaweicloud.com/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzf Python-${PYTHON_VERSION}.tgz \
    && rm Python-${PYTHON_VERSION}.tgz

# 编译安装
WORKDIR /tmp/Python-${PYTHON_VERSION}
RUN ./configure \
    --enable-shared \
    --prefix=/usr/local \
    --enable-optimizations \
    && make -j$(nproc) \
    && make install \
    && ldconfig

# 清理
RUN rm -rf /tmp/Python-${PYTHON_VERSION}

# 创建软链接
RUN ln -sf /usr/local/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip3

# 验证
RUN python3.11 --version && pip3.11 --version

WORKDIR /app

CMD ["/bin/bash"]