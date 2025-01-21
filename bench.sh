#!/bin/bash

cd cmake-build-debug

echo 'SSIM GPU'
time ./IQM --method SSIM --input ../input.png --ref ../ref.png --output ../out.png >/dev/null

echo 'SSIM CPU'
time ./IQM --method SSIM_CPU --input ../input.png --ref ../ref.png --output ../out.png >/dev/null