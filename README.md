# MNet: A Framework to Reduce Fruit Image Misclassification 
## original artcile is avaialble at https://www.iieta.org/journals/isi/paper/10.18280/isi.260203

This repositoy cosnists of 3 python files: 
1) FC_with_inceptionv3_plainModel.py
2) FC_with_inceptionv3_TL.py
3) FC_with_inceptionv3_MNet.py

## Fruit classification using pre-trained model:
To classify the fruits "inceptionv3" pre-trained model is used. you can refer the Keras Applications (https://keras.io/api/applications/) and can try the code with differnt pre-trained models.

## Dataset 
Sample images are used from the original dataset in this project. 
Datasets are published and available at:
1) FRUITSGB: TOP INDIAN FRUITS WITH QUALITY (link: https://ieee-dataport.org/open-access/fruitsgb-top-indian-fruits-quality )
    ( Cite as: Vishal Meshram, Koravat Thanomliang, Supawadee Ruangkan, Prawit Chumchu, Kailas Patil, July 8, 2020, "FruitsGB: Top Indian Fruits with quality ", IEEE Dataport, doi: https://dx.doi.org/10.21227/gzkn-f379.)
2) FruitNet: Indian Fruits Dataset with quality (Good, Bad & Mixed quality) (link: https://data.mendeley.com/datasets/b6fftwbr2v/2)
    (Cite as: PATIL, Kailas; MESHRAM, Vishal (2021), “FruitNet: Indian Fruits Dataset with quality (Good, Bad & Mixed quality)”, Mendeley Data, V2, doi: 10.17632/b6fftwbr2v.2)


# Code Files (.py programs)
### 1. FC_with_inceptionv3_plainModel
In this project weights of the inceptionv3 model are used. 4 layers are added at the end: global average pooling layer,  dense layer, dropout layer, and 1 softmax layer.

### 2. FC_with_inceptionv3_TL
Fruit classification with "Trasnfer Learning" technique. where 20% of layers are freezed and model is retrained on the dataset.

### 2. FC_with_inceptionv3_MNet
Fruit classification with "Trasnfer Learning" using MNet framework. we created separated models i.e. one for fruti classification and one for quality. in the last phase output is merged. 


### Hyperparameters tuning
Hyperparameters in Machine learning are those parameters that are explicitly defined by the user to control the learning process. some examles are: 1) Learning rate for training a neural network 2) Train-test split ratio 3) Batch Size 4) Number of Epochs. As per the rquirement user can chage these parameters in the code.

### Other related articles:
1) FruitNet: Indian fruits image dataset with quality for machine learning applications (link: https://www.sciencedirect.com/science/article/pii/S2352340921009616)
   (Cite as: Vishal Meshram, Kailas Patil, FruitNet: Indian fruits image dataset with quality for machine learning applications, Data in Brief, Volume 40, (2022), 107686, https://doi.org/10.1016/j.dib.2021.107686)
2) Machine learning in agriculture domain: A state-of-art survey (link: https://www.sciencedirect.com/science/article/pii/S2667318521000106)
   (Cite as: Vishal Meshram, Kailas Patil, Vidula Meshram, Dinesh Hanchate, S.D. Ramkteke, Machine learning in agriculture domain: A state-of-art survey, Artificial Intelligence in the Life Sciences, Volume 1, (2021), 100010, https://doi.org/10.1016/j.ailsci.2021.100010)

