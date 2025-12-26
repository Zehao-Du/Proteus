#!/bin/bash
# 将本地修改后的 vllm 代码直接集成到主仓库
# 使用方法: bash integrate_vllm.sh

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔄 开始集成本地 vllm 代码到主仓库...${NC}"
echo ""

# 检查 vllm 目录是否存在
if [ ! -d "vllm" ]; then
    echo -e "${RED}❌ 错误: vllm 目录不存在${NC}"
    exit 1
fi

# 检查是否在 git 仓库中
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ 错误: 当前目录不是 Git 仓库${NC}"
    exit 1
fi

# 备份当前分支
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${YELLOW}📋 当前分支: ${CURRENT_BRANCH}${NC}"

# 检查 vllm 是否有未提交的修改
cd vllm
if [ -d ".git" ]; then
    echo -e "${YELLOW}📦 检测到 vllm 是独立的 git 仓库${NC}"
    
    # 检查是否有未提交的修改
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then
        echo -e "${YELLOW}⚠️  vllm 有未提交的修改，建议先提交${NC}"
        read -p "是否继续？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # 显示 vllm 的当前状态
    echo -e "${GREEN}📊 vllm 当前状态:${NC}"
    git log --oneline -1 2>/dev/null || echo "   无法获取 git 信息"
    echo ""
fi
cd ..

# 创建备份
BACKUP_DIR="vllm_backup_$(date +%Y%m%d_%H%M%S)"
echo -e "${YELLOW}💾 创建备份: ${BACKUP_DIR}${NC}"
cp -r vllm "$BACKUP_DIR"
echo -e "${GREEN}✅ 备份完成${NC}"
echo ""

# 移除 vllm 的 .git（如果存在）
if [ -d "vllm/.git" ]; then
    echo -e "${YELLOW}🗑️  移除 vllm/.git 以将其转换为普通目录...${NC}"
    rm -rf vllm/.git
    echo -e "${GREEN}✅ 已移除${NC}"
    echo ""
fi

# 移除 vllm 的 .gitignore（可选，保留也可以）
# rm -f vllm/.gitignore

# 检查主仓库的 .gitignore
if grep -q "^vllm/" .gitignore 2>/dev/null || grep -q "^vllm$" .gitignore 2>/dev/null; then
    echo -e "${YELLOW}⚠️  检测到 .gitignore 中忽略了 vllm${NC}"
    echo "   需要从 .gitignore 中移除 vllm 的忽略规则"
    read -p "是否自动更新 .gitignore？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # 移除 vllm 相关的忽略规则
        sed -i '/^vllm\/$/d' .gitignore 2>/dev/null || true
        sed -i '/^vllm$/d' .gitignore 2>/dev/null || true
        echo -e "${GREEN}✅ 已更新 .gitignore${NC}"
    fi
    echo ""
fi

# 从 git 中移除旧的 vllm 引用（如果是 submodule）
if git ls-files --error-unmatch vllm > /dev/null 2>&1; then
    echo -e "${YELLOW}🗑️  移除旧的 vllm 引用...${NC}"
    git rm --cached vllm 2>/dev/null || true
    echo -e "${GREEN}✅ 已移除${NC}"
    echo ""
fi

# 添加 vllm 目录到 git
echo -e "${GREEN}📦 添加 vllm 目录到 git...${NC}"
git add vllm/

# 显示将要提交的文件
echo ""
echo -e "${GREEN}📋 将要提交的文件:${NC}"
git status --short vllm/ | head -n 20
TOTAL_FILES=$(git status --short vllm/ | wc -l)
echo -e "${YELLOW}   总计: ${TOTAL_FILES} 个文件${NC}"
echo ""

# 询问是否提交
read -p "是否现在提交？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 获取提交信息
    echo ""
    echo -e "${YELLOW}💬 请输入提交信息:${NC}"
    echo "格式: feat: 集成本地修改后的 vllm 代码"
    echo ""
    read -p "提交信息: " COMMIT_MSG
    
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG="feat: 集成本地修改后的 vllm 代码到主仓库"
    fi
    
    echo ""
    echo -e "${GREEN}💾 提交更改...${NC}"
    git commit -m "$COMMIT_MSG"
    
    echo ""
    echo -e "${GREEN}✅ 完成！${NC}"
    echo ""
    echo -e "${YELLOW}📝 下一步:${NC}"
    echo "   1. 推送到远程: git push origin ${CURRENT_BRANCH}"
    echo "   2. 或者创建 PR"
    echo ""
    echo -e "${YELLOW}💡 提示:${NC}"
    echo "   - 备份保存在: ${BACKUP_DIR}"
    echo "   - 如果出现问题，可以恢复: rm -rf vllm && mv ${BACKUP_DIR} vllm"
else
    echo ""
    echo -e "${YELLOW}📝 文件已添加到暂存区，但未提交${NC}"
    echo "   使用以下命令提交:"
    echo "   git commit -m 'feat: 集成本地修改后的 vllm 代码'"
fi

echo ""

