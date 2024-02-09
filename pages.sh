#!/bin/bash
set -eux -o pipefail

# Script to help publish a new build to a "pages" branch for https://craig-macomber.github.io/rusty-flame/

# Confirm no uncomitted changes exist:
# TODO: this fails to error in the case where there are new files.
git update-index --refresh
git diff-index --quiet HEAD --

# Require master branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "master" ]]; then
    echo "run this on the master branch"
    exit 1
fi

# Ensure tests pass
cargo test

# Web release build
wasm-pack build --target web

# Regenerate the pages branch from the current one
git branch -d pages
git checkout -b pages

git add -f pkg/rusty_flame_bg.wasm
git add -f pkg/rusty_flame.js

git commit -m "Web build for pages"
git push -f --set-upstream origin pages

git checkout $BRANCH

echo "Done Publishing"