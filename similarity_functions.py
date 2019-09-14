import cv2
import my_utils
import numpy as np
import colormoments
import math


def simiarity_sift():
    a = colormoments.getsift('test/')
    x = colormoments.getsift('test1/')

    T1= 0.71
    T2= 16
    a=a[0]
    len1=len(a)
    print(a)
    print(x)
    v=0
    count=0
    sum1=0
    sum2=0
    for b in x:
        len2 = len(b)
        #print(len1,len2)
        card_number = 0
        for i in range(len1):

            for j in range(len2):
                d_xy = math.sqrt((a[i][0]-b[j][0])*(a[i][0]-b[j][0]) + (a[i][1]-b[j][1])*(a[i][1]-b[j][1]))
                v1=a[i][4:]
                v2=b[j][4:]

                v = 0
                for v_ind in range(0, 128):
                    v = v + ((v1[v_ind] - v2[v_ind]) * (v1[v_ind] - v2[v_ind]))

                v=math.sqrt(v)
                #print(v)
                #print(d_xy)
                sum1 = sum1 + v
                sum2 = sum2 + d_xy
                if v<=T1 and d_xy<=T2 :
                    count=count+1
                    card_number = card_number+1

        print(sum1/(len1+len2) , sum2/(len1+len2))
        print((len1 + len2) / 2 - card_number)


def similarity_cmom():

    return


