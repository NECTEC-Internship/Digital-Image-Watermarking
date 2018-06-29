import cv2
import copy
import numpy as np
import haarPsi as hp
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import color
from PIL import Image, ImageFilter

red_patch = mpatches.Patch(color='red', label='Red channel')
green_patch = mpatches.Patch(color='green', label='Green channel')
blue_patch = mpatches.Patch(color='blue', label='Blue channel')

original_patch = mpatches.Patch(color='black', label='Original singular value')
watermarked_patch = mpatches.Patch(color='cyan', label='Watermarked singular value')
normalized_patch = mpatches.Patch(color='magenta', label='Normalized singular value')

def watermarkImage(image,left,right,embedded_bit):
    global LEFT
    global RIGHT
    LEFT = left
    RIGHT = right

    ORI_U = []
    ORI_S = []
    ORI_VH = []
    try:
        m,n,_ = image.shape
        c = 3
    except:
        m,n = image.shape
        c = 1

    for i in range(c):
        try:
            u,s,vh = np.linalg.svd(image[:,:,i],full_matrices=True)
        except:
            u,s,vh = np.linalg.svd(image[:,:],full_matrices=True)
        ORI_U.append(u)
        ORI_S.append(s)
        ORI_VH.append(vh)
        #plt.figure()
        #plt.plot(s,'o:')
    ############################################################
    # Embedding process
    S = np.copy(ORI_S)
    EMBEDDING_S = []
    for (i,s) in enumerate(S):
        s[left:right]=s[left] if embedded_bit else s[right]
        #plt.figure()
        #plt.plot(s,'o:')
        #plt.xlim([xmin,xmax])
        EMBEDDING_S.append(s)
    ############################################################
    # Reconstruct the watermarked image
    EMBEDDING_IMG = np.zeros(image.shape, dtype=float)
    #???????????????????????????????????????????????????????????
    '''
    for i in range(c):
        new_S = np.zeros((m,n))
        for j in range(min(m,n)):
            new_S[j,j]=S[0,j]
            try:
                EMBEDDING_IMG[:,:,i] = np.dot(ORI_U[i],np.dot(new_S,ORI_VH[i])).astype(float)
            except:
                EMBEDDING_IMG[:,:] = np.dot(ORI_U,np.dot(new_S,ORI_VH)).astype(float)

    return EMBEDDING_IMG
    '''
    try:
        # R channel
        new_S = np.zeros((m,n))
        for i in range(min(m,n)):
            new_S[i,i]=S[0,i]
        new_r = np.dot(ORI_U[0],np.dot(new_S,ORI_VH[0]))
        # G channel
        new_S = np.zeros((m,n))
        for i in range(min(m,n)):
            new_S[i,i]=S[1,i]
        new_g = np.dot(ORI_U[1],np.dot(new_S,ORI_VH[1]))
        # B channel
        new_S = np.zeros((m,n))
        for i in range(min(m,n)):
            new_S[i,i]=S[2,i]
        new_b = np.dot(ORI_U[2],np.dot(new_S,ORI_VH[2]))
        EMBEDDING_IMG[:,:,0] = new_r
        EMBEDDING_IMG[:,:,1] = new_g
        EMBEDDING_IMG[:,:,2] = new_b
    except:
        # Grayscale
        new_S = np.zeros((m,n))
        for i in range(min(m,n)):
            new_S[i,i]=S[0,i]
        EMBEDDING_IMG = np.dot(ORI_U[0],np.dot(new_S,ORI_VH[0]))

    EMBEDDING_IMG = normalizeImage(EMBEDDING_IMG)
    return EMBEDDING_IMG

def normalizeImage(input_image):
    normalized_image = copy.deepcopy(input_image)
    cv2.normalize(normalized_image,normalized_image,0,255,cv2.NORM_MINMAX,dtype=-1)
    return normalized_image.astype(np.uint8)

def extractImage(image_to_extract):
    global LEFT
    global RIGHT

    upper_bound = LEFT-1
    lower_bound = RIGHT+1
    try:
        m,n,_ = image_to_extract.shape
        c = 3
    except:
        m,n = image_to_extract.shape
        c = 1
    extracted_bits = []
    for i in range(c):
        try:
            _,extracted_s,_ = np.linalg.svd(image_to_extract[:,:,i],full_matrices=True)
        except:
            _,extracted_s,_ = np.linalg.svd(image_to_extract[:,:],full_matrices=True)
        middle_point = round(0.5*(upper_bound+lower_bound))
        middle_value = 0.5*(extracted_s[upper_bound]+extracted_s[lower_bound])
        #print(extracted_s[upper_bound]-extracted_s[middle_point],extracted_s[middle_point]-extracted_s[lower_bound], end=" ")
        if (extracted_s[upper_bound]-extracted_s[middle_point]) > (extracted_s[middle_point]-extracted_s[lower_bound]):
            extracted_bits.append(0)
        else:
            extracted_bits.append(1)
    return extracted_bits

