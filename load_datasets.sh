git clone https://github.com/oBoii/Unifying-Disentanglement-and-Concept-Based-Learning.git
cd Unifying-Disentanglement-and-Concept-Based-Learning
git clone https://github.com/google-deepmind/dsprites-dataset.git
python store_each_dsprite_as_seperate_file.py
python main.py

#
mkdir awa2-dataset && cd awa2-dataset && mkdir AwA2-data && cd AwA2-data
wget -O data.zip https://cvml.ista.ac.at/AwA2/AwA2-data.zip
unzip data.zip
cd .. # AwA2-data
cd .. # awa2-dataset
cd .. # Unifying-Disentanglement-and-Concept-Based-Learning
python resize_awa2.py

sudo apt update;
sudo apt install rsync;
# copy the remaining files in awa2-dataset/AwA2-data/Animals_with_Attributes2/ to Animals_with_Attributes2_resized/
# but exclude the JPEGImages folder
rsync -av --progress awa2-dataset/AwA2-data/Animals_with_Attributes2/ awa2-dataset/AwA2-data/Animals_with_Attributes2_resized/ --exclude=JPEGImages