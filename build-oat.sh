# make release directories
mkdir ./release
mkdir ./release/frameserve
mkdir ./release/framefilt
mkdir ./release/viewer
mkdir ./release/detector
mkdir ./release/posicom
mkdir ./release/decorator
mkdir ./release/posifilt 
mkdir ./release/recorder
mkdir ./release/positest

# Create makefiles and build
cd ./release/
cd ./frameserve
cmake ../../src/frameserve
make
cd ..

cd ./framefilt/
cmake ../../src/framefilt
make
cd ..

cd ./viewer/
cmake ../../src/viewer
make
cd ..

cd ./detector/
cmake ../../src/detector
make
cd ..

cd ./posicom/
cmake ../../src/posicom
make
cd ..

cd ./decorator/
cmake ../../src/decorator
make
cd ..

cd ./posifilt/ 
cmake ../../src/posifilt
make
cd ..

cd ./recorder/ 
cmake ../../src/recorder
make
cd ..

cd ./positest/ 
cmake ../../src/positest
make
cd ..

cd ..

# Move each component to the release dir 
mv ./release/frameserve/oat-frameserve  ./oat/libexec
mv ./release/frameserve/oat-calibrate 	./oat/libexec
mv ./release/framefilt/oat-framefilt 	./oat/libexec
mv ./release/viewer/oat-view 			./oat/libexec
mv ./release/detector/oat-detect 		./oat/libexec
mv ./release/posicom/oat-posicom 		./oat/libexec
mv ./release/decorator/oat-decorate 	./oat/libexec
mv ./release/posifilt/oat-posifilt 		./oat/libexec
mv ./release/recorder/oat-record 		./oat/libexec
mv ./release/positest/oat-positest 		./oat/libexec

