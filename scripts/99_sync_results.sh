#!/usr/bin/env bash
# Push results back to the private repo from Colab.
#
# ONLY the reports and the manifest go back. Never checkpoints, never the HF cache,
# never a token. .gitignore enforces the first two; the grep below is the last line of
# defence for the third.
set -euo pipefail
cd "$(dirname "$0")/.."

MSG="${1:-results: phase0 run}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Results arrive from a rented GPU box that was cloned fresh, which means it is sitting
# on the default branch. Pushing there sends unreviewed numbers straight to main and
# bypasses the PR where `make-report`'s verdict is actually read. A results branch costs
# one command and makes the review the default rather than the exception.
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
    SUGGESTED="results/$(date -u +%Y%m%d-%H%M)"
    echo "ABORT: on '$BRANCH'. Results go to a branch and through a PR, never straight"
    echo "       to the default branch — the gate verdict is reviewed there."
    echo
    echo "  git checkout -b $SUGGESTED"
    echo "  bash scripts/99_sync_results.sh \"$MSG\""
    echo
    echo "  (RDL_ALLOW_DEFAULT_BRANCH=1 overrides, for a repo with no PR flow.)"
    [ "${RDL_ALLOW_DEFAULT_BRANCH:-0}" = "1" ] || exit 1
    echo "RDL_ALLOW_DEFAULT_BRANCH=1 set — pushing to '$BRANCH' anyway."
fi

# `git add -A` does NOT add ignored files. Until ADR-0060 the only un-ignored thing under
# results/ was the manifest, so this line staged the manifest and left every report
# behind — and the next step was "destroy the instance". Belt and braces: stage, then
# CHECK that each report on disk actually reached the index, and refuse to continue if
# one did not. A silent omission here loses evidence that cost GPU hours.
git add -A results/ docs/ || true

MISSING=0
while IFS= read -r artifact; do
    [ -e "$artifact" ] || continue
    if ! git ls-files --error-unmatch "$artifact" >/dev/null 2>&1; then
        echo "NOT STAGED: $artifact"
        MISSING=1
    fi
done < <(
    {
        find results -maxdepth 1 -name 'REPORT*.md' -o -maxdepth 1 -name 'gate_verdict*.json' \
            -o -maxdepth 1 -name 'fig*.png'
        find results -mindepth 2 -maxdepth 2 \
            \( -name 'condition_report.json' -o -name 'repro_report.json' \
            -o -name 'measure_report.json' -o -name 'handoff_evidence.json' \)
    } 2>/dev/null
)
if [ "$MISSING" = "1" ]; then
    echo
    echo "ABORT: report artifacts exist on disk but are not in the index. They are being"
    echo "       ignored — check the results/ negation rules in .gitignore (ADR-0060)."
    echo "       Do NOT destroy this instance: the evidence is only here."
    exit 1
fi
if git diff --cached | grep -nE '(hf_[A-Za-z0-9]{34}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{36})'; then
    echo "ABORT: a credential pattern appears in the staged diff."
    git reset
    exit 1
fi

# Large artifacts must never enter history.
if git diff --cached --name-only | grep -Ei '\.(safetensors|bin|ckpt|pt|pth|gguf)$'; then
    echo "ABORT: model weights are staged. Check .gitignore."
    git reset
    exit 1
fi

if git diff --cached --quiet; then
    echo "nothing to commit"
    exit 0
fi

git commit -m "$MSG"
git push origin "$BRANCH"
git log --oneline -3
