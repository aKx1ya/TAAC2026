@echo off
chcp 65001 >nul

echo ========================================
echo 太棒了我又乱写了一点东西！！！
echo ========================================
echo.


set /p COMMIT_MSG="👉 请输入 commit 内容 (按回车继续): "


if "%COMMIT_MSG%"=="" (
  echo ❌ 错误: 提交信息不能为空！操作已取消。
  pause
  exit /b 1
)

echo.
echo 🔄 1/4 拉取远程最新代码 (git pull)...
git pull

echo.
echo 📦 2/4 暂存所有本地更改 (git add .)...
git add .

echo.
echo 📝 3/4 提交更改 (git commit)...
git commit -m "%COMMIT_MSG%"

echo.
echo 🚀 4/4 推送到远程仓库 (git push)...
git push

echo.
echo ✅ 一键提交完成！
echo 按任意键关闭........
echo.

pause