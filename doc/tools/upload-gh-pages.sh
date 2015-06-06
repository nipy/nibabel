#!/bin/bash
# Upload website to gh-pages
HTML_DIR=$1
if [ -z "$HTML_DIR" ]; then
    echo "Need to specify build directory"
    exit 1
fi
UPSTREAM_REPO=$2
if [ -z "$UPSTREAM_REPO" ]; then
    echo "Need to specify upstream repo"
    exit 1
fi
cd $HTML_DIR
git init
git checkout -b gh-pages
git add *
# A nojekyll file is needed to tell github that this is *not* a jekyll site:
touch .nojekyll
git add .nojekyll
git commit -a -m "Documentation build - no history"
git remote add origin $UPSTREAM_REPO
git push origin gh-pages --force
rm -rf .git  # Yes
