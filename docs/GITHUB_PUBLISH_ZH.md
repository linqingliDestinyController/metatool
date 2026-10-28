# 发布代码

在解压后的 `metatool` 目录中操作。先确认 README 的“仅在线 RL”范围符合你希望公开的方法；保留第三方许可证。模型和工作目录放在仓库之外。

```bash
git init -b main
git add .
git status --short
git commit -m "Release metatool online RL implementation"
gh auth login
gh repo create metatool --public --source=. --remote=origin --push
```

已有远端空仓库时，改用 `git remote add origin <仓库URL>` 和 `git push -u origin main`。不要覆盖已有同名远端仓库。

代码和配置修改后，可用 `python scripts/make_manifest.py` 更新完整性清单；已经开始训练的运行应继续绑定原代码，新的代码使用新的工作目录。
