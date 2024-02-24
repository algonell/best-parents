#!/bin/sh

echo "Maven..."
mvn -f ~/git/best-parents versions:use-latest-releases -DgenerateBackupPoms=false
find ~/git/best-parents -name "*.xml" -exec xmllint --format '{}' --output '{}' \;

echo "build..."
~/git/best-parents/build.sh
