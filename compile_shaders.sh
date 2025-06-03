#!/bin/bash

searchPath="*shaders/*"
files=`find $searchPath -type f`
dirs=`find $searchPath -type d`

rm -rf shaders_out
mkdir shaders_out
touch shaders_out/.gitkeep

# first create subfolders as needed
for i in $dirs; do
  path=${i#shaders/}
  mkdir -p "shaders_out/$path"
done

for i in $files; do
  # check if shader file contains "void main", if not, it's shared file with definitions, so don't compile it
  grep "void main" $i >> /dev/null
  if [[ $? = 1 ]]; then
    continue
  fi
  # remove suffix and prefix
  path=${i#shaders/}
  path=${path%.glsl}
  # compile shaders to files which are then included
  glslc "$i" -o "shaders_out/$path.inc" -mfmt=c --target-env=vulkan1.2
done
