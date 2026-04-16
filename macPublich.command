#!/bin/bash

# 切换到脚本所在的目录，确保在正确的 git 仓库中执行（Mac 双击运行必备）
cd "$(dirname "$0")" || exit

echo "========================================"
echo "太棒了我又乱写了一点东西！！！"
echo "========================================"
echo ""

# 读取用户输入
read -p "👉 请输入 commit 内容 (按回车继续): " COMMIT_MSG

# 判断输入是否为空 (-z 判断字符串长度是否为 0)
if [ -z "$COMMIT_MSG" ]; then
  echo "❌ 错误: 提交信息不能为空！操作已取消。"
  # 暂停一下让用户看到报错
  read -n 1 -s -r -p "按任意键退出........"
  echo ""
  exit 1
fi

echo ""
echo "🔄 1/4 拉取远程最新代码 (git pull)..."
git pull

echo ""
echo "📦 2/4 暂存所有本地更改 (git add .)..."
git add .

echo ""
echo "📝 3/4 提交更改 (git commit)..."
git commit -m "$COMMIT_MSG"

echo ""
echo "🚀 4/4 推送到远程仓库 (git push)..."
git push

echo ""
echo "✅ 一键提交完成！"
echo ""

# 相当于 Windows 的 pause
read -n 1 -s -r -p "按任意键关闭........"
echo ""