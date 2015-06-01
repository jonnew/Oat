# make release directories
mkdir ./release
mkdir ./release/frameserve/
mkdir ./release/framefilt/
mkdir ./release/viewer/
mkdir ./release/detector/
mkdir ./release/posicom/
mkdir ./release/decorator/
mkdir ./release/posifilt/ 
mkdir ./release/recorder

# Create makefiles and build
cd ./release/
cd ./frameserve/
cmake ../../frameserve
make
cd ..

cd ./framefilt/
cmake ../../framefilt
make
cd ..

cd ./viewer/
cmake ../../viewer
make
cd ..

cd ./detector/
cmake ../../detector
make
cd ..

cd ./posicom/
cmake ../../posicom
make
cd ..

cd ./decorator/
cmake ../../decorator
make
cd ..

cd ./posifilt/ 
cmake ../../posifilt
make
cd ..

cd ./recorder/ 
cmake ../../recorder
make
cd ..

cd ..

# Move each component to the release dir 
mv ./release/frameserve/oat-frameserve  ../sub/libexec
mv ./release/frameserve/oat-calibrate 	../sub/libexec
mv ./release/framefilt/oat-framefilt 	../sub/libexec
mv ./release/viewer/oat-viewer 			../sub/libexec
mv ./release/detector/oat-detector 		../sub/libexec
mv ./release/posicom/oat-posicom 		../sub/libexec
mv ./release/decorator/oat-decorate 	../sub/libexec
mv ./release/posifilt/oat-posifilt 		../sub/libexec
mv ./release/recorder/oat-record 		../sub/libexec

