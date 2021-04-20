# Convolutional Neural Network
#part 1
# Importing the libraries
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense

#initialising the cnn
classifier = Sequential()

#step 1 convolution
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),padding='same', activation="relu"))
#32 number of feature detecter/kernel with 3 rows and 3  columns
#input shape it is the shape of input shape in which we are going to apply our feature detector through the convolution operation.
#we are converting all our image to one same single format using input_shape. 
# 3 is number of channel and 64,64 are dimensions i.e pixels.
#Dense function is use to connect fully connected layer

#Step 2 Pooling 
#it reduces size of feature map/kernel
classifier.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

#adding 2nd convolutional layer
classifier.add(Convolution2D(32,3,3,padding='same',activation='relu'))
classifier.add(MaxPooling2D(pool_size=2,strides=2,padding='valid'))

#step 3 flattening
classifier.add(Flatten())

#step4 full connection

classifier.add(Dense(units =128 ,activation = 'relu'))

#output function
classifier.add(Dense(units =1 ,activation = 'sigmoid'))

#part 2 Fitting the cnn to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True

)

test_datagen = ImageDataGenerator(rescale=1./255) #It is use to preprocess he image from test set

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64,64),
                                            batch_size = 32,
                                            class_mode = 'binary')
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit(
        training_set,
        steps_per_epoch=8000//32, #len(training_set)//batch_size
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)
#validation_data corresponds to the dataset on which we wantt to evaluate performance of our cnn. here we use test set
# validation_steps corresponds to number of images in our test set
