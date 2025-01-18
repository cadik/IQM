#!/bin/bash

searchPath="*shaders/*"
files=`find $searchPath -type f`
dirs=`find $searchPath -type d`

# first create subfolders as needed
for i in $dirs; do
  path=${i#shaders/}
  mkdir -p "shaders_out/$path"
done

for i in $files; do
  # remove suffix and prefix
  path=${i#shaders/}
  path=${path%.glsl}
  # compile shaders
  glslc "$i" -o "shaders_out/$path.spv"
done
