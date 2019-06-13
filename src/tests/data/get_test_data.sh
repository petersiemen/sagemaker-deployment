#!/usr/bin/env bash

rm $(find aclImdb/ -type f)

cp $(find  ../../../data/aclImdb/train/pos/ -type f  | tail  -n 100) aclImdb/train/pos/
cp $(find  ../../../data/aclImdb/train/neg/ -type f  | tail  -n 100) aclImdb/train/neg/
cp $(find  ../../../data/aclImdb/test/pos/ -type f  | tail  -n 100) aclImdb/test/pos/
cp $(find  ../../../data/aclImdb/test/neg/ -type f  | tail  -n 100) aclImdb/test/neg/
