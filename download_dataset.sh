#!/bin/bash

url="https://drive.google.com/uc?export=download&id="

# make data folder
mkdir -p data
cd data

echo "#################### Downloading News Headline Generation dataset to /data ####################"
curl -c ./cookie -s -L $url"18cNTr5kLfCj4-LudLBvR2ch8m3ThgOEs"> /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=18cNTr5kLfCj4-LudLBvR2ch8m3ThgOEs" -o data.zip
echo "####################unzipping News Headline Generation dataset ####################"
unzip -q data.zip -x __MACOSX*
rm data.zip
rm cookie

