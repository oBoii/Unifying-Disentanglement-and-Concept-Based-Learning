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
