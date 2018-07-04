import math
import random
import cv2
import copy
import numpy as np
import haarPsi as hp
import matplotlib.image as img
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mode
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
    plot_style = ["ro:","go:","bo:"]
    try:
        m,n,_ = image_to_extract.shape
        c = 3
    except:
        m,n = image_to_extract.shape
        c = 1
    extracted_bits = []

    #plt.figure()
    #plt.title("Extracted singular value")
    for i in range(c):
        try:
            _,extracted_s,_ = np.linalg.svd(image_to_extract[:,:,i],full_matrices=True)
        except:
            _,extracted_s,_ = np.linalg.svd(image_to_extract[:,:],full_matrices=True)

        #plt.plot(extracted_s[upper_bound-10:lower_bound+10],plot_style[i])
        middle_point = round(0.5*(upper_bound+lower_bound))
        middle_value = 0.5*(extracted_s[upper_bound]+extracted_s[lower_bound])
        #print(extracted_s[upper_bound]-extracted_s[middle_point],extracted_s[middle_point]-extracted_s[lower_bound])
        if (extracted_s[upper_bound]-extracted_s[middle_point]) > (extracted_s[middle_point]-extracted_s[lower_bound]):
            extracted_bits.append(0)
        else:
            extracted_bits.append(1)

    extracted_bits = mode(extracted_bits, axis=None)

    return extracted_bits[0][0]

def watermarkImageBlock(image,left,right,bits_to_embed,block_size):
    global BLOCK_SIZE
    global N_BITS
    global WATERMARK_BITS

    BLOCK_SIZE = block_size
    N_BITS = len(bits_to_embed)
    WATERMARK_BITS = []
    try:
        m,n,_ = image.shape
        c = 3
    except:
        m,n = image.shape
        c = 1
    
    WATERMARKING_IMG = np.array(image, dtype=float)
    
    bit_n = 0
    for y in range(0,m,block_size):
        if y+block_size > m:
            break
        for x in range(0,n,block_size):
            if x+block_size > n:
                break
            for i in range(c):
                try:
                    WATERMARKING_IMG[y:y+block_size,x:x+block_size,i] = watermarkImage(image[y:y+block_size,x:x+block_size,i],left,right,bits_to_embed[bit_n])
                except:
                    WATERMARKING_IMG[y:y+block_size,x:x+block_size] = watermarkImage(image[y:y+block_size,x:x+block_size],left,right,bits_to_embed[bit_n])
           
            WATERMARK_BITS.append(bits_to_embed[bit_n])
            
            bit_n = bit_n+1
            if bit_n >= len(bits_to_embed):
                break
        if bit_n >= len(bits_to_embed):
            break
            
    return WATERMARKING_IMG.astype(np.uint8)

def extractImageBlock(image_to_extract):
    
    global BLOCK_SIZE
    global N_BITS
    global WATERMARK_BITS

    try:
        m,n,_ = image_to_extract.shape
        c = 3
    except:
        m,n = image_to_extract.shape
        c = 1
    
    EXTRACTING_IMG = copy.deepcopy(image_to_extract)
    
    extracted_bits = []
    bit_n = 0     
    
    for y in range(0,m,BLOCK_SIZE):
        if y+BLOCK_SIZE > m:
            break
        for x in range(0,n,BLOCK_SIZE):
            #plt.figure()
            if x+BLOCK_SIZE > n:
                break
            try:
                #plt.imshow(EXTRACTING_IMG[y:y+BLOCK_SIZE,x:x+BLOCK_SIZE,:])
                extracted_bit = extractImage(EXTRACTING_IMG[y:y+BLOCK_SIZE,x:x+BLOCK_SIZE,:])
            except:
                #plt.imshow(EXTRACTING_IMG[y:y+BLOCK_SIZE,x:x+BLOCK_SIZE],cmap='gray')
                extracted_bit = extractImage(EXTRACTING_IMG[y:y+BLOCK_SIZE,x:x+BLOCK_SIZE])
                
            extracted_bits.append(extracted_bit)

            bit_n = bit_n+1
            if bit_n >= N_BITS:
                break
        if bit_n >= N_BITS:
            break
    print("WATERMARKED_BITS\tEXTRACTED_BITS")
    for i in range(N_BITS):
        print("%d\t\t\t%d"%(WATERMARK_BITS[i],extracted_bits[i]))

    return ber(WATERMARK_BITS, extracted_bits)

def ber(watermarked_bits, extracted_bits):

    m = len(watermarked_bits)

    return sum(np.logical_xor(watermarked_bits,extracted_bits))/m

def psnr(img1, img2):
    mse = np.mean( np.square(img1 - img2) )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def experiment(input_image, left, right, block_size, num_bits, attack_func=None):
    
    experiment_image = copy.deepcopy(input_image)

    bits_to_watermark = [random.randint(0,1) for i in range(num_bits)]
    
    watermarked_image = watermarkImageBlock(experiment_image,left,right,bits_to_watermark,block_size)

    attacked_image = attack_func(watermarked_image) if attack_func is not None else watermarked_image

    return  extractImageBlock(attacked_image),hp.haar_psi(input_image,attacked_image)[0]