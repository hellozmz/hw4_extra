"""Setup the package."""
import os
import shutil
import sys
import sysconfig
import platform
import subprocess

from setuptools import find_packages
from setuptools.dist import Distribution


from setuptools import setup

__version__ = "0.1.0"

# 定义编译函数
def compile_with_make():
    try:
        # 切换到上一级目录
        os.chdir('..')
        # 执行 make 命令
        subprocess.run(['make'], check=True)
        # 切换回当前目录
        os.chdir('python')
    except subprocess.CalledProcessError as e:
        print(f"编译失败: {e}")
        sys.exit(1)

# 在 setup 之前执行编译操作
compile_with_make()

setup(
    name="needle",
    version=__version__,
    description="CMU-DLsys",
    zip_safe=False,
    packages=find_packages(),
    package_dir={"needle": "needle"},
    url="dlsyscourse.org"
)
