import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

class GANDataBalancer:
    def __init__(self, sampling_strategy=0.05, random_state=42, latent_dim=100, learning_rate=0.0001):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

    def build_generator(self, output_dim):
        model = Sequential([
            Dense(32, activation='relu', input_dim=self.latent_dim),
            Dense(64, activation='relu'),
            Dense(output_dim, activation='linear')
        ])
        return model

    def build_discriminator(self, input_dim):
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        # Use the custom Adam optimizer with the specified learning rate
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.latent_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        gan = Model(gan_input, gan_output)
        # Use the custom Adam optimizer with the specified learning rate
        optimizer = Adam(learning_rate=self.learning_rate)
        gan.compile(optimizer=optimizer, loss='binary_crossentropy')
        return gan

    def fit(self, X, y, epochs=10000, batch_size=64):
        # Identify minority and majority classes
        X_minority = X[y == 1]
        X_majority = X[y == 0]
        self.resample_number = int(self.sampling_strategy * len(X_majority) - len(X_minority))
        self.X_columns = X.columns
        minority_class_samples = X_minority.values

        self.generator = self.build_generator(X.shape[1])
        self.discriminator = self.build_discriminator(X.shape[1])
        self.gan = self.build_gan()

        for epoch in range(epochs):
            # Train Discriminator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_data = self.generator.predict(noise)
            real_data = minority_class_samples[
                np.random.randint(0, minority_class_samples.shape[0], size=batch_size)
            ]
            X_combined = np.vstack((real_data, fake_data))
            y_combined = np.hstack((np.ones(batch_size), np.zeros(batch_size)))
            self.discriminator.trainable = True
            d_loss = self.discriminator.train_on_batch(X_combined, y_combined)

            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            y_gen = np.ones(batch_size)
            self.discriminator.trainable = False
            g_loss = self.gan.train_on_batch(noise, y_gen)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}/{epochs} | Discriminator Loss: {d_loss} | Generator Loss: {g_loss}")

    def resample(self, X, y):
        synthetic_data = self.generator.predict(
            np.random.normal(0, 1, (self.resample_number, self.latent_dim))
        )
        synthetic_df = pd.DataFrame(synthetic_data, columns=X.columns)
        balanced_X = pd.concat([X, synthetic_df])
        balanced_y = pd.concat([y, pd.Series([1] * synthetic_df.shape[0])])
        return balanced_X, balanced_y

    def fit_resample(self, X, y, epochs=1000, batch_size=64):
        self.fit(X, y, epochs=epochs, batch_size=batch_size)
        return self.resample(X, y)
