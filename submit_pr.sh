#!/bin/bash
# 便捷的 PR 提交脚本
# 使用方法: bash submit_pr.sh [分支名] [提交信息]

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查是否在 git 仓库中
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ 错误: 当前目录不是 Git 仓库${NC}"
    exit 1
fi

# 获取当前分支或使用参数
BRANCH_NAME="${1:-feat/update-$(date +%Y%m%d)}"
COMMIT_MSG="${2:-}"

echo -e "${GREEN}📋 检查当前状态...${NC}"
git status

echo ""
echo -e "${GREEN}🌿 创建/切换到分支: ${BRANCH_NAME}${NC}"
if git show-ref --verify --quiet refs/heads/"$BRANCH_NAME"; then
    echo -e "${YELLOW}   分支已存在，切换到该分支${NC}"
    git checkout "$BRANCH_NAME"
else
    git checkout -b "$BRANCH_NAME"
fi

echo ""
echo -e "${GREEN}📦 查看修改的文件...${NC}"
git status --short

echo ""
read -p "是否添加所有修改的文件? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git add .
else
    echo "请手动添加文件: git add <file1> <file2> ..."
    exit 1
fi

echo ""
if [ -z "$COMMIT_MSG" ]; then
    echo -e "${YELLOW}💾 请输入提交信息（多行，以空行结束）:${NC}"
    echo "格式: <type>: <subject>"
    echo "      <空行>"
    echo "      <body>"
    echo ""
    echo "类型: feat, fix, docs, style, refactor, test, chore"
    echo ""
    COMMIT_MSG=$(cat)
fi

if [ -z "$COMMIT_MSG" ]; then
    echo -e "${RED}❌ 错误: 提交信息不能为空${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}💾 提交更改...${NC}"
echo "$COMMIT_MSG" | git commit -F -

echo ""
echo -e "${GREEN}🚀 推送到 origin...${NC}"
git push -u origin "$BRANCH_NAME"

echo ""
echo -e "${GREEN}✅ 完成！${NC}"
echo ""
echo -e "${YELLOW}📝 下一步：${NC}"
echo "   1. 访问你的 Fork: https://github.com/$(git config user.name)/eBPF-TokenFlow"
echo "   2. 点击 'Compare & pull request' 按钮"
echo "   3. 或者直接访问:"
echo "      https://github.com/Zehao-Du/eBPF-TokenFlow/compare/main...$(git config user.name):eBPF-TokenFlow:${BRANCH_NAME}"
echo ""
echo "   4. 填写 PR 描述并提交"
