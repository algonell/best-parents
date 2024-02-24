#!/bin/sh

clear

echo
echo "google-java-format..."
java -jar ~/Misc/google-java-format-*.jar --replace $(git ls-files ~/git/best-parents/*.java)

echo
echo "SpotBugs..."
mvn -f ~/git/best-parents spotbugs:check
