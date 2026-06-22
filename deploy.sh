#!/usr/bin/env bash
# ============================================
# 校园智能助手 — 一键部署脚本 (Ubuntu)
# 用法: bash deploy.sh
# ============================================
set -euo pipefail

echo "========================================"
echo "  校园智能助手 部署脚本"
echo "========================================"

# 1. 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "[1/6] 安装 Docker..."
    apt-get update && apt-get install -y docker.io docker-compose-plugin
    systemctl enable docker && systemctl start docker
else
    echo "[1/6] Docker 已安装 ✓"
fi

# 2. 检查 .env
if [ ! -f .env ]; then
    echo "[2/6] 创建 .env 文件（请在其中填入你的 API Key）..."
    cat > .env << 'ENVEOF'
# DeepSeek API 配置（将 your_key_here 替换为你的真实 API Key）
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek/deepseek-chat
ENVEOF
    echo "⚠️  请编辑 .env 文件，填入你的 OPENAI_API_KEY，然后重新运行此脚本"
    exit 1
fi
echo "[2/6] .env 文件已存在 ✓"

# 3. 加载环境变量
echo "[3/6] 加载环境变量..."
set -a
source .env
set +a

# 4. 检查 API Key
if [ "${OPENAI_API_KEY:-}" = "" ] || [ "${OPENAI_API_KEY:-}" = "your_key_here" ]; then
    echo "❌ 请在 .env 文件中填入有效的 OPENAI_API_KEY"
    exit 1
fi
echo "[4/6] API Key 已配置 ✓"

# 5. 构建并启动容器
echo "[5/6] 构建 Docker 镜像并启动服务..."
docker compose build
docker compose up -d

echo "[6/6] 等待服务就绪..."
sleep 5

# 6. 健康检查
if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo ""
    echo "========================================"
    echo "  ✅ 部署成功！"
    echo ""
    echo "  后端 API:   http://101.37.125.208:8000"
    echo "  API 文档:   http://101.37.125.208:8000/docs"
    echo "  健康检查:   http://101.37.125.208:8000/health"
    echo ""
    echo "  查看日志:   docker compose logs -f"
    echo "  停止服务:   docker compose down"
    echo "========================================"
else
    echo "❌ 服务启动失败，查看日志: docker compose logs"
    exit 1
fi
