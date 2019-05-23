from segmentation_models import Unet

from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2gray
import numpy as np
import cv2
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


TRAIN_DATASET_PATH='/home/rbuddhad/NIH-XRAY/train'
VALID_DATASET_PATH='/home/rbuddhad/NIH-XRAY/validation'
IMAGE_SIZE    = (256, 256)
CROP_LENGTH   = 836
NUM_CLASSES   = 2
BATCH_SIZE    = 16  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 3
WEIGHTS_FINAL = 'model-cropped-final.h5'
NUMBER=0

def searchForPatches(filename,lines,index):

	for i in range(len(lines)):
		line=lines[i][0:-1]
		fields=line.split('#')
		if(filename == fields[0]):
			line=lines[i+index][0:-1]
			fields=line.split('#')
			return fields

	fields=['',0,0,155,155]		
	return fields
			
def random_crop(img, random_crop_size, filename,index,lines):
    fields=searchForPatches(filename,lines,index)
    x=0
    y=0
    dx=0
    dy=0

    x=int(fields[1])
    y=int(fields[2])
    dx=int(fields[3])
    dy=int(fields[4])
    img=img[y:(y+dy), x:(x+dx), :]
    img = cv2.resize(img,(224,224))
    img=img/255.0
    return img


def crop(img, random_crop_size):
    height, width = img.shape[0], img.shape[1]
    dy0, dx0 = 836,836
    x0 = 94
    y0 = 45
    img=img[y0:(y0+dy0), x0:(x0+dx0), :]
    img=img/255
    img = cv2.resize(img,(224,224))
    
    return img

def crop_generator(batches, crop_length,lines):#224

    filenames=((batches.filenames))
    while True:
	    batch_x= next(batches)
	    batch_crops_inp = np.zeros((4,batch_x.shape[0], 224, 224,3))#224
	    for i in range(batch_x.shape[0]):            
	        for j in range(4):
	            batch_crops_inp[j][i] = random_crop(batch_x[i], (crop_length, crop_length),filenames[i],j,lines)
	    batch_crops_inp=np.reshape(batch_crops_inp,(batch_crops_inp.shape[0]*batch_crops_inp.shape[1],224,224,3))
	    batch_crops_out=batch_crops_inp

	    batch_crops_inp=rgb2gray(batch_crops_inp)
	    batch_crops_inp=np.reshape(batch_crops_inp,(batch_crops_inp.shape[0],224,224,1))

	    yield(batch_crops_out,batch_crops_inp)


def main():


	with open('/home/rbuddhad/NIH-XRAY/train_sml.txt') as f1:
		lines1 = f1.readlines()

	with open('/home/rbuddhad/NIH-XRAY/validation_sml.txt') as f2:
		lines2 = f2.readlines()			


	train_datagen = ImageDataGenerator()
	train_batches = train_datagen.flow_from_directory(TRAIN_DATASET_PATH,
                                                  target_size=(1024,1024),
                                                  shuffle=True,
                                                  class_mode=None,
                                                  batch_size=BATCH_SIZE)

	valid_datagen = ImageDataGenerator()
	valid_batches = valid_datagen.flow_from_directory(VALID_DATASET_PATH ,
    	                                              target_size=(1024,1024),
    	                                              shuffle=False,
        	                                          class_mode=None,
        	                                          batch_size=BATCH_SIZE)
	
	train_crops_orig = crop_generator(train_batches, CROP_LENGTH,lines1) #224
	valid_crops_orig = crop_generator(valid_batches, CROP_LENGTH,lines2)

	
	model = Unet(backbone_name='resnet18', encoder_weights=None)
	model.load_weights('best_model.h5')
	model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['mae','mean_squared_error'])
	model.summary()

	callbacks = [EarlyStopping(monitor='val_loss', patience=10),
	             ModelCheckpoint(filepath='best_model1.h5', monitor='val_loss', save_best_only=True),
	             TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)]
	model.fit_generator(generator=train_crops_orig,
                    steps_per_epoch=100,
                    validation_data=valid_crops_orig,
                    callbacks=callbacks,
                    validation_steps=200,
                    epochs=1000,
					shuffle=True)
	model.save('unet1.h5')




if __name__=="__main__":
	main()



