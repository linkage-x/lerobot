#!/bin/bash

read -r -p "请输入你购买的秘钥: " api_key

CODEX_DIR="$HOME/.codex"
mkdir -p "$CODEX_DIR"

# 写入配置文件
config_toml_path="$CODEX_DIR/config.toml"
auth_json_path="$CODEX_DIR/auth.json"

cat > "$auth_json_path" << 'EOF'
{
  "OPENAI_API_KEY": null
}
EOF

cat > "$config_toml_path" << 'EOF'
model_provider = "codex"
model = "gpt-5"
model_reasoning_effort = "high"
disable_response_storage = true

[model_providers.codex]
name = "codex"
base_url = "https://cc.585dg.com/codex/v1"
wire_api = "responses"
env_key = "CODEX_API_KEY"
EOF

# 智能检测用户使用的shell并写入相应的配置文件
current_shell=$(basename "$SHELL")

case "$current_shell" in
    "bash")
        echo "检测到bash shell，写入到 ~/.bashrc"
        echo "export CODEX_API_KEY=$api_key" >> ~/.bashrc
        source ~/.bashrc 2>/dev/null || true
        ;;
    "zsh")
        echo "检测到zsh shell，写入到 ~/.zshrc"
        echo "export CODEX_API_KEY=$api_key" >> ~/.zshrc
        source ~/.zshrc 2>/dev/null || true
        ;;
    *)
        echo "检测到其他shell ($current_shell)，同时写入到 ~/.bashrc 和 ~/.zshrc"
        echo "export CODEX_API_KEY=$api_key" >> ~/.bashrc
        echo "export CODEX_API_KEY=$api_key" >> ~/.zshrc
        ;;
esac

echo "配置完成！请重新打开终端或运行 'source ~/.${current_shell}rc' 使环境变量生效。"