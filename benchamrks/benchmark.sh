#!/bin/bash

#echo "$1"
echo "Method: $2"

#find src iamges
refs=`find src_images -type f | grep "ref"`

for ref in $refs; do
  echo $ref
  inp=${ref%ref.*}test.png
  find $inp 2> /dev/null >> /dev/null
  if [[ $? = 1 ]]; then
      inp=${inp%.png}.jpg
  fi

  out=`$1 --method $2 -i 50 --input $inp --ref $ref -v`
  echo -n "    "
  echo "$out" | grep "VRAM" | head -n 1
  echo -n "    "
  echo "$out" | grep "Median" | head -n 1
done