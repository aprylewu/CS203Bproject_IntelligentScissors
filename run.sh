#!/bin/bash
# 创建编译输出目录
mkdir -p build/classes
# 编译 Java 文件
javac -d build/classes -cp "lwjgl-release-3.3.6-custom/*:algs4/lib/*" src/*.java
# 运行程序，传入图片路径参数
java -XstartOnFirstThread -cp "build/classes:lwjgl-release-3.3.6-custom/*:algs4/lib/*" -Djava.library.path=lwjgl-release-3.3.6-custom Main desk.png 