#!/bin/bash
# Script to update the build system documentation based on current Makefiles

set -e  # Exit on error

echo "Extracting targets from Makefiles..."

# Extract PHONY targets from top-level Makefile
ROOT_PHONY=$(grep -A1 "^\.PHONY:" Makefile | tr -d '\n' | sed 's/\.PHONY://' | tr -s ' ' | sed 's/ \\ / /g')
echo "Top-level Makefile targets: $ROOT_PHONY"

# Extract PHONY targets from alphago_demo Makefile
if [ -f alphago_demo/Makefile ]; then
    ALPHAGO_PHONY=$(grep -A1 "^\.PHONY:" alphago_demo/Makefile | tr -d '\n' | sed 's/\.PHONY://' | tr -s ' ')
    echo "AlphaGo Makefile targets: $ALPHAGO_PHONY"
fi

# Extract PHONY targets from golang_implementation Makefile
if [ -f golang_implementation/Makefile ]; then
    GOLANG_PHONY=$(grep -A1 "^\.PHONY:" golang_implementation/Makefile | tr -d '\n' | sed 's/\.PHONY://' | tr -s ' ')
    echo "Golang Makefile targets: $GOLANG_PHONY"
fi

# Extract PHONY targets from cpp_implementation Makefile
if [ -f cpp_implementation/Makefile ]; then
    CPP_PHONY=$(grep -A1 "^\.PHONY:" cpp_implementation/Makefile | tr -d '\n' | sed 's/\.PHONY://' | tr -s ' ')
    echo "C++ Makefile targets: $CPP_PHONY"
fi

# Update version info in the build_system.md
TODAY=$(date +%Y-%m-%d)
VERSION_LINE="| $(date +%Y.%m.%d) | $TODAY | Updated documentation with latest targets |"

echo "Updating docs/build_system.md with latest version info: $VERSION_LINE"

# Check if we need to create or update the version table
if grep -q "## 13. Version History" docs/build_system.md; then
    # Add the new version to the existing version history table
    sed -i "" -e "/## 13. Version History/,/^$/s/|---------|------|---------|/|---------|------|---------|\\n$VERSION_LINE/" docs/build_system.md
else
    echo "Warning: Could not find version history section in docs/build_system.md"
    echo "Please add the new version manually: $VERSION_LINE"
fi

echo "Documentation update complete. Please review changes to docs/build_system.md."
echo "Consider running './test_build_system.sh' to verify build system functionality." 