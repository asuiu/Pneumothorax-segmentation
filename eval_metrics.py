import keras.backend as K
def Rsqured(PA,GA): 
    PA = K.batch_flatten(PA)   #it flattens each data samples of a batch
    GA = K.batch_flatten(GA)
    
    GT = (PA1 ∩ GA)
    PS = (PA2 ∩ GA)
    num = num ∩ num 

    overleap(PA, GA) = 2 * |PA∩GA| / min(|PA|,|GA|)

    return num/overleap
        
    #K.mean(features) 
    # 3 VARIABLES( categ: ce e comun, cat e detectat de noi, cat nu e detectat de noi)
    # FP nu au nici o area (pneumothorax)
    # classifiy about category: dupa ce rulam reteaua, sa clasificam imag in fct de cate zone de pneumotorax  
