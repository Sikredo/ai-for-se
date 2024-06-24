# AI-For-SE

## Running the application
1) Download cleansed BigVul dataset (from Moodle) and put .json file in "bigvul-data" directly
2) Install necessary libraries and dependencies (transformers, sklearn, torch, numpy, ...)
3) Run create_embeddings.py to pre-compute the context embeddings for the training and testing data set (those are saved locally in two .pt files)
4) Run classifier_training.py to train, test and evaluate the classification model

### For detailed documentation see Project Report