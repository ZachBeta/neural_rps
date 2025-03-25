#!/bin/bash

# Add all changes to the staging area
git add .

# Commit the changes with a message
# pass args through to git commit
# default to "update" if no args
git commit -m "${1:-update}"