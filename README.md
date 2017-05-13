# Image-Captioning-with-keras
An Image Caption generator based on google's Show and Tell (2015) paper

## Dependencies

- keras (2.0) [use tensorflow backend]
- tensorflow (1.0)

## Running the code
Executing the Final+Model.py file will run 100 epochs over the whole MS COCO dataset. The data can be downloded by running .sh scripts provided in dataset folder. If you don't have a GPU the computation might take too long. You can disable callbacks (for logging tensorboard files and checkpoints to save weights) to reduce computation time. For disabling callbacks just remove callback parameter from the call model.fit().

## Files:
- **Final model training** : HTML file downloaded from actual .ipynb file after training on complete MS COCO dataset. This file includes training loss and validation loss after each epochs. Total Epochs around 130 and maximum accuracy 98.64. Time taken to train the model -> approx 7 hours.
- **Final Model** : Final code where predictions are made for images in validation data. Probabitlies are also computed for all words in a caption(caption_length = 17). Predicted and Actual capiton is displayed alongside each other just below the funciton call. Image and its corresponding 5 captions are also displayed.
- Final+Model.py: Python code for FinalModel.ipynb notebook
- GPUstats: .png image file displaying a screenshot while training RNN on gpu. Can give an idea on GPU memory, GPU graphics and CPU usage on training time.

## GPU used for training

Single Nvidia GTX 970

## Reserch Paper Reffered

Show and Tell: A Neural Image Caption Generator
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
(Submitted on 17 Nov 2014, last revised 20 Apr 2015)
