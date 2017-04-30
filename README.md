# Image-Captioning-with-keras
An Image Caption generator based on google's Show and Tell (2015) paper

## CAUTION
This work is currently under progress that is why the code is little scattered. We hope to get the things done in a few weeks and upload
properly modularized code by then.

## Files:
- **Final model training** : HTML file downloaded from actual .ipynb file after training on complete MS COCO dataset. This file includes training loss and validation loss after each epochs. Total Epochs around 130 and maximum accuracy 98.64. Time taken to train the model -> approx 7 hours.
- **Fianl Model** : Final code where predictions are made for images in validation data. Probabitlies are also computed for all words in a caption(caption_length = 17). Predicted and Actual capiton is displayed alongside each other just below the funciton call. Image and its corresponding 5 captions are also displayed.

## Dependencies

- keras (2.0) [use tensorflow backend]
- tensorflow (1.0)

## GPU used for training

Single Nvidia GTX 970

## Reserch Paper Reffered

Show and Tell: A Neural Image Caption Generator
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
(Submitted on 17 Nov 2014, last revised 20 Apr 2015)
