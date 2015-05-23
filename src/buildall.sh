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
mkdir bin
mv ./release/frameserve/frameserve ./bin
mv ./release/frameserve/calibrate ./bin
mv ./release/framefilt/framefilt ./bin
mv ./release/viewer/viewer ./bin
mv ./release/detector/detector ./bin
mv ./release/posicom/posicom ./bin
mv ./release/decorator/decorate ./bin
mv ./release/posifilt/posifilt ./bin
mv ./release/recorder/record ./bin

#make -C ./camserve/bin
#make -C ./backsubtractor/bin
#make -C ./viewer/bin/
#make -C ./detector/bin
#make -C ./posicom/bin
#make -C ./decorator/bin
#make -C ./posifilt/bin
#make -C ./recorder/bin
