# Fraud-Detection-Using-Autoencoder-and-Variational-Autoencoder
### Link for Dataset: https://drive.google.com/file/d/18SiycZ5Q9-le4hifS-FFuxFzbJIFtKk0/view?usp=sharing

This project aims to detect fraudulent transactions using both Autoencoder and VAE models. We imported the card transaction dataset using pandas and performed several preprocessing steps, including applying logarithmic transformations to 'Time' and 'Amount', and normalizing all other features using Min-Max scaling. The data was split into training and testing sets for both fraudulent and non-fraudulent transactions.

For the Autoencoder, we designed a sequential model with two dense layers in both the encoder and decoder, using ReLU activation for the hidden layers and Sigmoid activation for the output. We compiled the model using the Adam optimizer and Mean Squared Error (MSE) as the loss function.

For the Variational Autoencoder (VAE), we built a similar architecture, but with an encoder that reduces dimensionality in two stages, followed by a decoder that upscales it back to the original size. The VAE was also compiled using the Adam optimizer and MSE loss function.
