#!/bin/bash
# 提交 PR 到 upstream 仓库的脚本

set -e

# REPO_DIR="/home/v-boxiuli/Smart-Network-Diagnostic-System-powered-by-eBPF"
# cd "$REPO_DIR"

echo "📋 检查当前状态..."
git status

echo ""
echo "🌿 创建新分支..."
git checkout -b feat/model-optimization 2>/dev/null || git checkout feat/model-optimization

echo ""
echo "📦 添加修改的文件..."
git add dashboard.py train_model.py smart_agent.py run_experiment.sh model_result.png

echo ""
echo "💾 提交更改..."
git commit -m "feat: 优化模型训练和推理流程，增强特征工程

- dashboard.py: 支持加载包含scaler的模型bundle，扩展特征维度至5个
- train_model.py: 简化训练流程，统一保存模型和scaler字典格式
- smart_agent.py: 修复eBPF程序编译问题，添加必要头文件
- run_experiment.sh: 增强实验脚本，添加hint_server和llm_simulator支持
- 更新模型结果可视化图片"

echo ""
echo "🚀 推送到 origin..."
git push -u origin feat/model-optimization

echo ""
echo "✅ 完成！"
echo ""
echo "📝 下一步："
echo "   1. 访问 https://github.com/lbx154/Smart-Network-Diagnostic-System-powered-by-eBPF"
echo "   2. 点击 'Compare & pull request' 按钮"
echo "   3. 将 base repository 改为 Zehao-Du/eBPF-TokenFlow"
echo "   4. 或者直接访问: https://github.com/Zehao-Du/eBPF-TokenFlow/compare/main...lbx154:Smart-Network-Diagnostic-System-powered-by-eBPF:feat/model-optimization"

