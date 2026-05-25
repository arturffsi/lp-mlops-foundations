# Git Quick Guide

## Install & one-time setup

```bash
# macOS
brew install git

# Tell git who you are (shows up in every commit)
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## The 6 things you actually need to know

| Term | What it is | Command |
|------|------------|---------|
| **clone** | Download a repo to your laptop | `git clone <url>` |
| **branch** | A parallel line of work that doesn't touch `main` until you're ready | `git checkout -b feature/my-thing` |
| **commit** | A snapshot of your changes saved *locally* — like a save point | `git add .` then `git commit -m "msg"` |
| **push** | Upload your local commits to GitHub | `git push` |
| **pull** | Download the latest commits from GitHub into your local copy | `git pull` |
| **PR** (pull request) | Ask the team to review and merge your branch into `main` | Open it on GitHub's website |

---

## Cloning this repo

```bash
git clone https://github.com/arturffsi/lp-mlops-foundations.git
cd lp-mlops-foundations
```

## The everyday loop

```bash
# 1. Start fresh from main
git checkout main
git pull

# 2. Make a branch for your work
git checkout -b feature/add-eda-exercise

# 3. ...edit files in your editor...

# 4. Save your progress (one or more commits)
git add .                            # stage every change
git commit -m "Add EDA exercise"     # snapshot it

# 5. Send your commits to GitHub
git push -u origin feature/add-eda-exercise
# (the -u is only needed the first time you push the branch)

# 6. Open a Pull Request on GitHub
#    GitHub will print a URL right after `git push` — click it.
```

After someone reviews and approves your PR, click **Merge** on GitHub. Your branch's commits become part of `main`.

---

## Mental model

```
main           o───o───o───────────o   (shared history; protected)
                        \         /
your branch              o───o───o     (your work; merges in via PR)
                         ↑   ↑   ↑
                         commits
```

- **Local changes** live on your laptop until you `push`.
- **`main` is shared**. Never commit directly to it — always branch + PR.
- **A commit isn't on GitHub** until you `push`. A `push` isn't merged into `main` until the PR is approved.

---

## Common rescue commands

| You want to... | Command |
|---------------|---------|
| See what changed | `git status` (which files) / `git diff` (the actual lines) |
| Undo *unstaged* edits in one file | `git restore <file>` |
| Pull the latest `main` into your branch | `git checkout main && git pull && git checkout - && git merge main` |
| See recent history | `git log --oneline -10` |
| Throw away the last commit (kept locally) | `git reset --soft HEAD~1` |

---

## Tips

- **Commit messages**: short imperative — *"Fix Redshift auth"*, not *"Fixed the bug"*.
- **Commit often**: small commits are easier to review and revert than giant ones.
- **Never `git push --force`** to `main`. To your own branch is fine; to shared branches, ask first.
- **Never commit secrets** (passwords, API keys, `.env` files). Once pushed, they're in the history forever — rotate them.
