#!/bin/bash
# Update header-only libs

mkdir temp

# Clone latest RapidJSON
mkdir rapidjson
git clone https://github.com/miloyip/rapidjson.git temp/rapidjson/
cp -r temp/rapidjson/include/rapidjson/* ./rapidjson/

# Clone latest cpptoml
mkdir cpptoml
git clone https://github.com/skystrife/cpptoml.git temp/cpptoml/
cp -r temp/cpptoml/include/* ./cpptoml/

# Get latest catch 'compiled' header
#mkdir catch
#cd temp
#wget https://raw.githubusercontent.com/philsquared/Catch/master/single_include/#catch.hpp
# mv catch.hpp ../catch
#cd ..

rm -r temp

