#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# setup_github.sh  —  One-time GitHub push for rag-pipeline
# Usage: bash setup_github.sh YOUR_GITHUB_USERNAME
# ─────────────────────────────────────────────────────────────

set -e

USERNAME=${1:-"YOUR_USERNAME"}
REPO="rag-pipeline"

echo "👤 GitHub username: $USERNAME"
echo "📦 Repo name      : $REPO"
echo ""

# 1. Init git
git init
git add .
git commit -m "feat: initial RAG pipeline — FAISS + LangChain + Groq"

# 2. Create repo on GitHub (requires GitHub CLI: https://cli.github.com)
if command -v gh &>/dev/null; then
  gh repo create "$REPO" --public --source=. --remote=origin --push
  echo "✅ Repo created and pushed!"
else
  echo "ℹ️  GitHub CLI not installed."
  echo "   Manual steps:"
  echo "   1. Create a new repo at https://github.com/new  (name: $REPO)"
  echo "   2. Run:"
  echo "      git remote add origin https://github.com/$USERNAME/$REPO.git"
  echo "      git branch -M main"
  echo "      git push -u origin main"
fi
