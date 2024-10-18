#!/bin/bash

searchPath="*shaders/*"

for i in $searchPath; do
  # remove suffix and prefix
  path=${i#shaders/}
  path=${path%.glsl}
  # compile shaders
  glslc "$i" -o "shaders_out/$path.spv"
done
