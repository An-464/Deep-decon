#Pix2PixGAN: Brownlee, J. Generative Adversarial Networks with Python: Deep Learning Generative Models for Image Synthesis and Image Translation. (Machine Learning Mastery, 2019).

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from skimage import io # reading in stacks of .tif files 
from tifffile import imsave, imwrite
import GAN_model_1Step_Included
import GAN_model_2Step_Included
import shutil
import time

#Functions for creation, deleting folders/files
def CreateFolder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def DeleteFolder(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def DeleteFile(path):
    if os.path.exists(path):
        os.remove(path)

#Several functions to load images & rescale them to range [-1,1]
def load_images1(path_Gauss, path_Airy, files_Gauss, files_Airy):        
    for i in range(len(files_Airy)):
        x_train=io.imread(path_Airy+files_Airy[i])
        x_train=(x_train-127.5)/127.5 
        y_train=io.imread(path_Gauss+files_Gauss[i])
        y_train=(y_train-127.5)/127.5
        if i==0:
            x_train_ges=x_train
            y_train_ges=y_train
        else:       
            x_train_ges=np.append(x_train_ges,x_train, axis=0)
            y_train_ges=np.append(y_train_ges,y_train, axis=0)
    return x_train_ges, y_train_ges

def load_images2(path_Airy, files_Airy):
    x_train= io.imread(path_Airy+files_Airy)
    x_train=(x_train-127.5)/127.5
    return x_train

def load_images(path_Gauss, path_Airy):
    x_train=io.imread(path_Airy)
    x_train=(x_train-127.5)/127.5 
    y_train=io.imread(path_Gauss)
    y_train=(y_train-127.5)/127.5    
    return x_train, y_train

def load_images2_2(path_Airy):
    x_train= io.imread(path_Airy)
    x_train=(x_train-127.5)/127.5
    return x_train

#Return Status update to GUI
def Status_Update(root, Status, text):
    Status.config(text="                                           ")
    root.update_idletasks()
    Status.config(text=text)
    root.update_idletasks()

#Crop images to smaller sections for Step 1 of GAN training
def CropImages(path_Gauss, path_Airy, folder_save_Gauss, folder_save_Airy, root, Status):   
    Airy_Stack = path_Airy
    Gauss_Stack = path_Gauss
    L=int(np.shape(Airy_Stack)[0]/256)
    M=int(np.shape(Airy_Stack)[1]/256)
    N=int(np.shape(Airy_Stack)[2]/256)
    for k in range(L):
        for i in range(M):
            for j in range(N):
                Airy_small=Airy_Stack[k*256:(k+1)*256,i*256:(i+1)*256,j*256:(j+1)*256]
                #Airy_small1=Airy_small*255.0/float(np.max(Airy_small))
                Gauss_small=Gauss_Stack[k*256:(k+1)*256,i*256:(i+1)*256,j*256:(j+1)*256]
                #Gauss_small1=Gauss_small*255.0/float(np.max(Gauss_small))
                imwrite(folder_save_Airy+"%02.i_%02.i_%02.i.tif" %(k,i,j), Airy_small.astype('uint8'))
                imwrite(folder_save_Gauss+"%02.i_%02.i_%02.i.tif" %(k,i,j), Gauss_small.astype('uint8'))
    return L, M, N

#Transpose Stacks from xy to yz orientation
def Transpose(path, path_save, root, Status):
    Status_Update(root, Status, "Transposing started")
    files = os.listdir(path)
    for i in range(len(files)):
        stack= io.imread(path+files[i])
        #stack_new=np.transpose(stack,axes=[2, 0, 1])   #z to xy
        stack_new=np.transpose(stack,axes=[1, 2, 0])    #xy to z
        imwrite(path_save+files[i], stack_new.astype('uint8'))

#Create Gauss images with same sice like merged Airy images      
def MergeGauss(path, path_save, L, M, N):
    files = os.listdir(path)
    Merge_final=np.zeros((L*256,M*256,N*256))
    for k in range(L):
        for i in range(M):
            for j in range(N):
                Stack=io.imread(path+str(k).zfill(2)+"_"+str(i).zfill(2)+"_"+str(j).zfill(2)+".tif")
                Merge_final[k*256:(k+1)*256,i*256:(i+1)*256,j*256:(j+1)*256]=Stack
    imwrite(path_save, Merge_final.astype('uint8'))


# train pix2pix models
def train_S1(d_model, g_model, gan_model, dataset, n_epochs, n_batch, modelname, root, Status):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples 
        [X_realA, X_realB], y_real = GAN_model_1Step_Included.generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = GAN_model_1Step_Included.generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples       
        d_loss1, acc1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples       
        d_loss2, acc2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        #Give Status update
        if (i+1) % 5==0:
            text=str(i+1)+ '/'+str(n_steps)
            Status_Update(root, Status, text)
            #summarize_performance(i, g_model, dataset)
            #g_model.save_weights("model_Bicubic_tiff_256_NeuronData2_Epoch%s.h5" %(i+1))
    g_model.save_weights(modelname)


#Train Step 1 (yz orientation)
def Step1(path_Gauss_z, path_Airy_z, modelname, n_batch, n_epoch, root, Status):
    image_shape = (256,256,1)
    Status_Update(root, Status, "Step1 started")
    d_model = GAN_model_1Step_Included.define_discriminator(image_shape)
    g_model = GAN_model_1Step_Included.define_generator(image_shape)
    gan_model = GAN_model_1Step_Included.define_gan(g_model, d_model, image_shape)

    files_Gauss=os.listdir(path_Gauss_z)
    files_Airy=os.listdir(path_Airy_z)
    dataset = load_images1(path_Gauss_z, path_Airy_z, files_Gauss, files_Airy)
    Status_Update(root, Status, "Data loaded (S1)")
    train_S1(d_model, g_model, gan_model, dataset, n_epoch, n_batch, modelname, root, Status)
    return g_model

#Apply model to training data
def ApplyStep1(g_model, path_Airy_z, modelname, path_save_1):
    files_Airy=os.listdir(path_Airy_z)
    for i in range(len(files_Airy)):
        dataset = load_images2(path_Airy_z, files_Airy[i])
        g_model.load_weights(modelname)
        X = g_model.predict(dataset)
        Generated_Stack=X[:,:,:,0]
        Generated_Stack=Generated_Stack* 127.5 + 127.5
        imwrite(path_save_1+files_Airy[i], Generated_Stack.astype('uint8'))









#Transpose training data back to xy orientation, merge cropped stacks
def Transpose_Merge(path_save_1, path_save_xy, path_save_Merged1, L, M, N):
    files = os.listdir(path_save_1)

    for i in range(len(files)):
        stack= io.imread(path_save_1+files[i])
        stack_new=np.transpose(stack,axes=[2, 0, 1])   #z to xy
        #stack_new=np.transpose(stack,axes=[1, 2, 0])    #xy to z
        imwrite(path_save_xy+files[i], stack_new.astype('uint8'))
    # Merge_Stacks:
    files = os.listdir(path_save_xy)
    Merge_final=np.zeros((L*256,M*256,N*256))

    for k in range(L):
        for i in range(M):
            for j in range(N):
                Stack=io.imread(path_save_xy+str(k).zfill(2)+"_"+str(i).zfill(2)+"_"+str(j).zfill(2)+".tif")
                Merge_final[k*256:(k+1)*256,i*256:(i+1)*256,j*256:(j+1)*256]=Stack
    imwrite(path_save_Merged1, Merge_final.astype('uint8')) 


# train pix2pix models
def train_S2(d_model, g_model, gan_model, dataset, n_epochs, n_batch, modelname2, root, Status):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    start = time.time()
    for i in range(n_steps):
        # select a batch of real samples 
        [X_realA, X_realB], y_real = GAN_model_2Step_Included.generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake =GAN_model_2Step_Included.generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples       
        d_loss1, acc1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples       
        d_loss2, acc2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # give Status Update
        if (i+1) % 500==0:
            t = time.time()
            date=str(time.ctime(start+(t-start)*n_steps/(i+1)))
            text=str(i+1)+ '/'+str(n_steps)+"Ready: "+date
            Status_Update(root, Status, text)
    g_model.save_weights(str(modelname2))

#Train Step 2 (xy orientation)
def Step2(path_Gauss, path_Airy, modelname2, n_epoch, n_batch, root, Status):
    image_shape = (2048,2048,1)
    d_model = GAN_model_2Step_Included.define_discriminator(image_shape)
    g_model = GAN_model_2Step_Included.define_generator(image_shape)
    gan_model = GAN_model_2Step_Included.define_gan(g_model, d_model, image_shape)
    #gan_model.summary()
    dataset = load_images(path_Gauss, path_Airy)
    Status_Update(root, Status, "Data S.2 loaded")
    train_S2(d_model, g_model, gan_model, dataset, n_epoch, n_batch, modelname2, root, Status)
  
#Apply final model to training data
def ApplyStep2(g_model, path_Airy, modelname2, path_save_result):
    dataset = load_images2_2(path_Airy)
    Generated_Stack=np.zeros(np.shape(dataset))
    Steps=10
    g_model.load_weights(modelname2)
    for i in range(0, np.shape(dataset)[0], Steps):
        Generated_Stack[i:(i+Steps),:,:] = g_model.predict(dataset[i:(i+Steps),:,:])[:,:,:,0]
    Generated_Stack=Generated_Stack* 127.5 + 127.5
    imwrite(path_save_result, Generated_Stack.astype('uint8'))
    



