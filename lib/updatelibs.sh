#!/bin/bash
# Update header-only libs

mkdir temp
git clone git@github.com:miloyip/rapidjson.git temp/rapidjson/
cp -r temp/rapidjson/include/rapidjson/* ./rapidjson/
git clone git@github.com:skystrife/cpptoml.git temp/cpptoml/
cp -r temp/cpptoml/include/* ./cpptoml/
rm -r temp

