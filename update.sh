#!/bin/sh

echo "Maven..."
mvn -f ~/git/best-parents versions:use-latest-releases

echo "build..."
~/git/best-parents/build.sh
