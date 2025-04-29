import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train / 255.0  # Normalize

# Define the generator
def build_generator():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=100),
        layers.Dense(784, activation='sigmoid'),
        layers.Reshape((28, 28))
    ])
    return model

# Define the discriminator
def build_discriminator():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Build the GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential([generator, discriminator])
    return model

# Initialize GAN components
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Compile the models
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Function to train the GAN
def train_gan(epochs=10000, batch_size=32):
    for epoch in range(epochs):
        noise = tf.random.normal([batch_size, 100])
        generated_images = generator(noise)
        real_images = x_train[:batch_size]

        labels_real = tf.ones((batch_size, 1))
        labels_fake = tf.zeros((batch_size, 1))

        # Train discriminator
        with tf.GradientTape() as tape:
            real_loss = discriminator(real_images)
            fake_loss = discriminator(generated_images)
            loss = tf.reduce_mean(labels_real * real_loss + labels_fake * fake_loss)
        grads = tape.gradient(loss, discriminator.trainable_variables)
        discriminator.optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # Train generator via the GAN
        noise = tf.random.normal([batch_size, 100])
        with tf.GradientTape() as tape:
            fake_loss = discriminator(generator(noise))
            gen_loss = tf.reduce_mean(labels_real * fake_loss)
        grads = tape.gradient(gen_loss, generator.trainable_variables)
        generator.optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Generator Loss: {gen_loss.numpy()} | Discriminator Loss: {loss.numpy()}")

        if epoch % 5000 == 0:
            plt.imshow(generated_images[0], cmap='gray')
            plt.show()

train_gan(epochs=10000)
