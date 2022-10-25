from PIL import ImageTk, Image
import tkinter as tk
from skimage import io, restoration
import time
import numpy as np
from tifffile import imsave, imwrite
from tkinter import ttk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import GAN_model_2Step_Included
import time
import Perform_Training as PT
import tkinter as tk
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

#Find Maximum Intensity value in Stack to calculate contrast
def FindMaxima(beads,ListeMax):
    while (len(ListeMax)<50):
        Maximum=np.max(beads)
        Positionen=np.where(beads==Maximum)
        if Maximum>65000:
           beads[Positionen]=0 
        if Maximum<65000:
            for i in range(np.shape(Positionen)[1]):
                ListeMax.append(Maximum)
            beads[Positionen]=0
    return(ListeMax)

#Find Minimum Intensity value in Stack to calculate contrast
def FindMinima(beads,ListeMin):
    while (len(ListeMin)<50):
        Minimum=np.min(beads)
        Positionen=np.where(beads==Minimum)
        if (Minimum==0):
            beads[Positionen]=50000
        if (Minimum!=0):
            for i in range(np.shape(Positionen)[1]):
                ListeMin.append(Minimum)
            beads[Positionen]=50000
    return(ListeMin)

#Calculate Weber Contrast from Lists with Maxima/Minima
def CalculateContrast(beads):
    ListeFinalMax=np.zeros((100))
    ListeFinalMin=np.zeros((100))
    for i in range(100):
        ListeMax=[]
        ListeFinalMax[i]=np.average(FindMaxima(beads[i,:,:],ListeMax))
        ListeMin=[]
        ListeFinalMin[i]=np.average(FindMinima(beads[i,:,:],ListeMin))
    Contrast=(np.average(ListeFinalMax)-np.average(ListeFinalMin))/(np.average(ListeFinalMin))
    #print(Contrast)
    return Contrast

#Load images for Evaluation
def load_images2(Pfad2):
    x_train= io.imread(Pfad2)
    #IM_MAX= np.max(x_train, axis=0)
    return x_train

#Post Status update in GUI
def Status_Update(text):
    Status.config(text="                                           ")
    root.update_idletasks()
    Status.config(text=text)
    root.update_idletasks()

#Bicubic upscaling of Airy original Stack
def Bicubic_upscaling():
    Status_Update("Upscaling Started")
    Path_Airy=B2.get()
    Path_Airy_save=B3.get()
    stack= io.imread(Path_Airy)
    AnzahlBilder=np.shape(stack)[0]
    Stack_upscaled=np.zeros((AnzahlBilder, 2*np.shape(stack)[1],2*np.shape(stack)[2]))
    for i in range(AnzahlBilder): 
        Stack_upscaled[i,:,:]=cv2.resize(stack[i,:,:], (2*np.shape(stack)[1],2*np.shape(stack)[2]), interpolation=cv2.INTER_CUBIC)
    Stack_upscaled_Norm=Stack_upscaled*255/np.max(Stack_upscaled)
    imsave(Path_Airy_save, Stack_upscaled_Norm.astype('uint8'))
    Status_Update("Upscaling Finished")

#Function for Image normalization
def norm(matrix):
    normalized=matrix/abs(np.max(matrix))
    return normalized

#Richardson Lucy Deconvolution
def PerformRLDecon():
    Status_Update("RL Decon started")
    path_stack=RL1.get()
    path_save=RL2.get()
    path_PSF=RL3.get()

    #Import stack
    stack= io.imread(path_stack)
    stack=norm(stack)
   
    #Import PSF
    PSF_final= io.imread(path_PSF)
    #PSF_final=np.transpose(PSF_final,axes=[1, 2, 0])   #z to xy
    #PSF_final=np.uintc(PSF_final*255/np.max(PSF_final))
    PSFLarge=np.zeros(np.shape(stack))

    half0=int(np.shape(stack)[0]/2)
    half1=int(np.shape(stack)[1]/2)
    half2=int(np.shape(stack)[2]/2)
    PSFLarge[half0-30:half0+30,half1-40:half1+40,half2-40:half2+40]=PSF_final
    PSFLarge=norm(PSFLarge)

    stack_decon= restoration.richardson_lucy(stack, PSFLarge,5)
    stack_decon=255*stack_decon/np.max(stack_decon)
    imsave(path_save, stack_decon.astype('uint8'))
    Status_Update("RL Finished")

#Show Maximum intensity projection from Richardson-Lucy deconvolution
def show_image():
    imagefile= RL1.get()
    imagefile3= RL2.get() 
    
    IM=io.imread(imagefile)
    IM_MAX= np.max(IM, axis=0)
    filename_new="MAX_Stack.jpg"
    imwrite(filename_new, IM_MAX.astype('uint8'))
    MIP=Image.open(filename_new)
    image_resized=MIP.resize((500,500))
    image = ImageTk.PhotoImage(image_resized)
    imagebox1.config(image=image)
    imagebox1.image = image 

    IM=io.imread(imagefile3)
    IM_MAX= np.max(IM, axis=0)
    filename_new="MAX_Stack_decon.jpg"
    imwrite(filename_new, IM_MAX.astype('uint8'))
    MIP=Image.open(filename_new)
    image_resized=MIP.resize((500,500))
    image = ImageTk.PhotoImage(image_resized)
    imagebox3.config(image=image)
    imagebox3.image = image    

#GAN training
def Start_training():
    #Read in Data paths, Epoch numbers
    Status_Update("GAN training started")
    Airy_Stack1=A1.get()
    Gauss_Stack1=G2.get()
    E1=M1.get()
    Epoch1=int(E1)
    E2=M3.get()
    Epoch2=int(E2)
    modelname2=M2.get()
    
    Airy_Stack = io.imread(Airy_Stack1)
    Gauss_Stack = io.imread(Gauss_Stack1)

    #Create folders and files
    Folder="./Intermediate_Files/"
    PT.CreateFolder(Folder)
    folder_save_airy_small=PT.CreateFolder(os.path.join(Folder, "Airy_small/"))
    folder_save_gauss_small=PT.CreateFolder(os.path.join(Folder,"Gauss_small/"))
    folder_save_airy_small_z=PT.CreateFolder(os.path.join(Folder,"Airy_small_z/"))
    folder_save_gauss_small_z=PT.CreateFolder(os.path.join(Folder,"Gauss_small_z/"))
    Gauss_small_merged=os.path.join(Folder,"Gauss_small_merged.tif")
    folder_Result_Step1_z=PT.CreateFolder(os.path.join(Folder,"ResultStep1_z/"))
    folder_Result_Step1_xy=PT.CreateFolder(os.path.join(Folder,"ResultStep1_xy/"))
    Airy_Merged_Step1=os.path.join(Folder,"Airy_Step1_merged.tif")


    #_________________________________Crop_________________________________
  
    LMN=PT.CropImages(Gauss_Stack, Airy_Stack, folder_save_gauss_small, folder_save_airy_small, root, Status)

    #________________________________Transpose_____________________________

    #Transpose Airy+Gauss
    PT.Transpose(folder_save_airy_small, folder_save_airy_small_z, root, Status)
    PT.Transpose(folder_save_gauss_small, folder_save_gauss_small_z, root, Status)
    PT.MergeGauss(folder_save_gauss_small, Gauss_small_merged, LMN[0], LMN[1], LMN[2])
    Status_Update("Preparation Step1 finished")

    #________________________________STEP 1________________________________

    n_epoch=Epoch1
    n_batch=1
    modelname="./Intermediate_Files/Test.h5"
    g_Model=PT.Step1(folder_save_gauss_small_z, folder_save_airy_small_z, modelname, n_batch, n_epoch, root, Status)
    Status_Update("Step1 finished")

    # _____________________________Apply model 1___________________________

    PT.ApplyStep1(g_Model, folder_save_airy_small_z, modelname, folder_Result_Step1_z)

    #___________________________Transpose / Merge__________________________

    PT.Transpose_Merge(folder_Result_Step1_z, folder_Result_Step1_xy, Airy_Merged_Step1, LMN[0], LMN[1], LMN[2])
    Status_Update("Preparation Step2 finished")

    #_______________________________STEP 2_________________________________

    n_epoch=Epoch2
    n_batch=1
    PT.Step2(Gauss_small_merged, Airy_Merged_Step1, modelname2, n_epoch, n_batch, root, Status)

    # _____________________Apply model 2 to Training_______________________

    #PT.ApplyStep2(GAN_model_2Step_Included.define_generator((2048,2048,1)), Airy_Merged_Step1, modelname2, Airy_GANOut_Step1and2)

    if (cb.get()==1):
        PT.DeleteFolder(Folder)
    Status_Update("GAN training finished")

#Apply GAN model to Test Stack
def Apply_GAN():
    Status_Update("GAN Application Started")
    Airy_Stack1=A2.get()
    modelname2=ModelEntry2.get()
    Airy_GANOut_OnlyStep2=Save3.get()
    PT.ApplyStep2(GAN_model_2Step_Included.define_generator((2048,2048,1)), Airy_Stack1, modelname2, Airy_GANOut_OnlyStep2)
    Status_Update("GAN Application Finished")

#Evaluation SSIM+PSNR+Contrast
def Evaluation():
    Status_Update("Evaluation started")
    Gauss=E1.get()
    Airy=E2.get()
    GAN=E3.get()
 
    Stack_HR=load_images2(Gauss)
    Stack_LR=load_images2(Airy)
    Stack_GAN_Step2=load_images2(GAN)
    Anzahl=np.shape(Stack_HR)[2]

    MSE_LR=np.zeros((Anzahl))
    MSE_Step2=np.zeros((Anzahl))
    s_LR=np.zeros((Anzahl))
    s_Step2=np.zeros((Anzahl))

    for i in range(Anzahl):
        MSE_LR[i]=mse(Stack_HR[:,:,i],Stack_LR[:,:,i])
        MSE_Step2[i]=mse(Stack_HR[:,:,i],Stack_GAN_Step2[:,:,i])
        s_LR[i]= ssim(Stack_HR[:,:,i],Stack_LR[:,:,i])
        s_Step2[i]=ssim(Stack_HR[:,:,i],Stack_GAN_Step2[:,:,i])
    MSE_LR_ges=np.average(MSE_LR)
    MSE_Step2_ges=np.average(MSE_Step2)
    s_LR_ges=np.average(s_LR)
    s_Step2_ges=np.average(s_Step2)

    Contrast_Gauss=CalculateContrast(Stack_HR)
    Contrast_Airy=CalculateContrast(Stack_LR)
    Contrast_Step2=CalculateContrast(Stack_GAN_Step2)
    tk.Label(root, text="MSE:                   Airy: {}            GAN: {}".format(MSE_LR_ges,MSE_Step2_ges)).grid(row=i+1, column=11) 
    tk.Label(root, text="SSIM:                  Airy: {}            GAN: {}".format(s_LR_ges,s_Step2_ges)).grid(row=i+2, column=11) 
    tk.Label(root, text="Contrast: Gauss: {}    Airy: {}            GAN: {}".format(Contrast_Gauss,Contrast_Airy, Contrast_Step2)).grid(row=i+3, column=11) 
    Status_Update("Evaluation Finished")   




root = tk.Tk()
root.title("-- Deep Decon --")

#Create an Entry+Label for GUI at defined row/column
def LabelEntry(row, column, textLabel, textEntry, text):
    tk.Label(root, text=textLabel).grid(row=row, column=column)
    B2= tk.Entry(root, width=30, text=text)
    i=row+1
    B2.grid(row=i, column=column)
    B2.insert(0,textEntry) 
    i=i+1
    tk.Label(root, text=" ").grid(row=i, column=0)
    i=i+1
    return B2, i

#Create Entry+Label for GUI with defined columnspan
def TrainLabelEntry(row, column, textLabel, textEntry, text, columnspan):
    tk.Label(root, text=textLabel).grid(row=row, column=column, columnspan=3)
    B2= tk.Entry(root, width=30, text=text)
    i=row+1
    B2.grid(row=i, column=column, columnspan=3)
    B2.insert(0,textEntry) 
    i=i+1
    tk.Label(root, text=" ").grid(row=i, column=0)
    i=i+1
    return B2, i

#Create Gaps for easier overview
tk.Label(root, text=" ").grid(row=0, column=0)
tk.Label(root, text=" ").grid(row=14, column=4)
tk.Label(root, text=" ").grid(row=20, column=12)

#Bicubic____________________________________________________________________________________________________

i=1
tk.Label(root, text="Status update:",font=('Helvetica', 11, 'bold'), bg="yellow").grid(row=i, column=1)   
i=i+1
Status=tk.Label(root)
Status.grid(row=i, column=1) 
i=i+2
tk.Label(root, text="Bicubic upscale", font=('Helvetica', 11, 'bold')).grid(row=i, column=1)  
i=i+2

B2,i=LabelEntry(row=i, column=1, textLabel="Airy Original Stack", textEntry="Data/Airy_Org.tif", text="enter11")
B3,i=LabelEntry(row=i, column=1, textLabel="Save Result under:", textEntry="Data/Bicubic_Upscaled_Airy.tif", text="enter12")

StartButton = tk.Button(root, text='Start', padx=50, command=Bicubic_upscaling).grid(row=i, column=1)
ttk.Separator(master=root,orient='vertical').grid(row=1, column=2, rowspan=12, ipady=160, padx=20, pady=10)


#Train GAN____________________________________________________________________________________________________

i=1
tk.Label(root, text="Train GAN", font=('Helvetica', 11, 'bold')).grid(row=i, column=3, columnspan=3)  
i=i+2

A1,i=TrainLabelEntry(row=i, column=3, textLabel="Airy Stack (bicubic upscaled)", textEntry="Data/Bicubic_Upscaled_Airy.tif", text="enter7", columnspan=3)
G2,i=TrainLabelEntry(row=i, column=3, textLabel="Gauss Stack", textEntry="Data/Gauss.tif", text="enter8", columnspan=3)
M2,i=TrainLabelEntry(row=i, column=3, textLabel="Model name (.h5 format)", textEntry="Data/model.h5", text="enter16", columnspan=3)

StartButton = tk.Button(root, text='Start', padx=50, command=Start_training).grid(row=i, column=3, columnspan=3)

i=i+2
tk.Label(root, text="Epoch numbers (Step 1/2):").grid(row=i, column=3, columnspan=3)
M1= tk.Entry(root, width=10, text="enter9")
i=i+1
M1.grid(row=i, column=3)
M1.insert(0,"20")
M3= tk.Entry(root, width=10, text="enter17")
M3.grid(row=i, column=5)
M3.insert(0,"200")

#Want to delete intermediate files? --> Option here
i=i+1
cb=tk.IntVar()
C1=tk.Checkbutton(root, text="Delete intermediate files", variable=cb, onvalue=1, offvalue=0)
C1.grid(row=i, column=3, columnspan=3)

ttk.Separator(master=root,orient='vertical').grid(row=1, column=6, rowspan=12, ipady=160, padx=20, pady=10)

#Apply GAN_________________________________________________________________________________________________

i=1
tk.Label(root, text="Apply GAN", font=('Helvetica', 11, 'bold')).grid(row=i, column=7)  
i=i+2

A2,i=LabelEntry(row=i, column=7, textLabel="Airy Stack", textEntry="Data/Bicubic_Upscaled_Airy.tif", text="enter4")
ModelEntry2,i=LabelEntry(row=i, column=7, textLabel="Model name", textEntry="Data/model.h5", text="enter6")
Save3,i=LabelEntry(row=i, column=7, textLabel="Save GAN Result under:", textEntry="Data/Airy_GAN_Out.tif", text="enter5")

StartButton = tk.Button(root, text='Start', padx=50, command=Apply_GAN).grid(row=i, column=7)
ttk.Separator(master=root,orient='vertical').grid(row=1, column=8, rowspan=12, ipady=160, padx=20, pady=10)

#RL____________________________________________________________________________________________________

i=1
tk.Label(root, text="Richardson Lucy Decon.", font=('Helvetica', 11, 'bold')).grid(row=i, column=9)  
i=i+2

RL1,i=LabelEntry(row=i, column=9, textLabel="Stack path", textEntry="Data/Bicubic_Upscaled_Airy.tif", text="enter")
RL3,i=LabelEntry(row=i, column=9, textLabel="Enter PSF path", textEntry="Data/PSF.tif", text="enter3")
RL2,i=LabelEntry(row=i, column=9, textLabel="Save RL Result under:", textEntry="Data/Airy_RL_Deconvolved.tif", text="enter2")

StartButton = tk.Button(root, text='Start', padx=50, command=PerformRLDecon).grid(row=i, column=9)
ttk.Separator(master=root,orient='vertical').grid(row=1, column=10, rowspan=12, ipady=160, padx=20, pady=10)


#SHOW MIPS

i=i+3
frame = tk.Frame(root)
frame.grid(row=i, column=9)
other = tk.Button(frame, text="Show MIPs", padx=50, command=show_image)
other.pack(side=tk.LEFT)

# label to show the image
i=i+1
imagebox1 = tk.Label(root)
imagebox1.grid(row=i, column=0, columnspan=6)

imagebox3 = tk.Label(root)
imagebox3.grid(row=i, column=8, columnspan=7)


#Evaluation____________________________________________________________________________________________________

i=1
tk.Label(root, text="Perform Evaluation", font=('Helvetica', 11, 'bold')).grid(row=i, column=11)  
i=i+2

E1,i=LabelEntry(row=i, column=11, textLabel="Gauss Stack", textEntry="Data/Gauss.tif", text="enter13")
E2,i=LabelEntry(row=i, column=11, textLabel="Compare: e.g. RL Stack", textEntry="Data/Airy_RL_Deconvolved.tif", text="enter14")
E3,i=LabelEntry(row=i, column=11, textLabel="Compare: e.g. GAN Result", textEntry="Data/Airy_GAN_Out.tif", text="enter15")

StartButton = tk.Button(root, text='Start', padx=50, command=Evaluation).grid(row=i, column=11)
        
    
root.mainloop()

