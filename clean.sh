#!/bin/bash
# 删除编译输出目录
rm -rf build/
rm -rf bin/

# 删除源目录中的 .class 文件
find src -name "*.class" -type f -delete

# 删除 IDE 相关文件
rm -rf .idea/
rm -rf .vscode/
rm -f *.iml
rm -f .project
rm -f .classpath

# 删除临时目录
rm -rf lwjgl-temp/

# 删除系统临时文件
find . -name ".DS_Store" -delete
find . -name "Thumbs.db" -delete

# 删除其他临时文件
find . -name "*.log" -delete
find . -name "*.tmp" -delete
find . -name "*.bak" -delete

echo "项目已清理完成" 