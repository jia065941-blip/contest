# -*- coding: utf-8 -*-#
import os
import shutil
from setuptools import setup, find_packages
from Cython.Build import cythonize

PACKAGE_NAME = "envengine"
VERSION = "1.0.0"

# 查找所有 .py 文件（除了 __init__.py）
py_files = []
for root, dirs, files in os.walk(PACKAGE_NAME):
    for f in files:
        if f.endswith('.py') and f != '__init__.py':
            py_files.append(os.path.join(root, f))

print(f"找到 {len(py_files)} 个文件需要编译")

if os.path.exists('build_temp'):
    print("清理旧的 build_temp...")
    shutil.rmtree('build_temp')

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description="Your package description",
    author="Your Name",
    packages=find_packages(include=['envengine', 'envengine.*']),
    ext_modules=cythonize(
        py_files,
        build_dir="build_temp",
        language_level=3,
        compiler_directives={
            'boundscheck': True,
            'wraparound': True,
            'nonecheck': True,
        }
    ),

    package_data={
        PACKAGE_NAME: [
            '**/__init__.py',
            '**/*.so',
            '**/*.pyd',
        ],
    },
    include_package_data=True,

    install_requires=[
        "dataclasses-json==0.6.7",
        "paho-mqtt==2.1.0",
        "pygame==2.6.1",
        "pyproj==3.7.2",
        "requests==2.34.2",
        "colorlog==6.10.1",
        "Cython==3.2.6",
        "fastdisjointset==1.0.3",
        "numpy==2.4.6",
        "websockets==16.0"
    ],

    python_requires="==3.11.*",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ]
)
