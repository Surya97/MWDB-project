import cv2
import my_utils
import numpy as np
import featuresextraction
import math
import collections


def simiarity_sift(source, loaded_imgs, k):
    a = source
    T1= 0.71
    T2= 16
    a=a[0]
    len1=len(a)
    simiarity_measure_map = {}
    v=0
    sum1=0
    sum2=0
    #print(len1)
    for image_id, b in loaded_imgs.items():
        len2 = len(b)

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

                sum1 = sum1 + v
                sum2 = sum2 + d_xy

                if v<=T1 and d_xy<=T2 :
                    card_number = card_number+1
        simiarity_measure_map[image_id] = (len1 + len2) / 2 - card_number

    sorted_simiarity_measure_map = collections.OrderedDict(
        sorted(simiarity_measure_map.items(), key=lambda measure_val: measure_val[1]))

    count = 0

    for image_id, similarity_measure_val in sorted_simiarity_measure_map.items():
        if count == int(k) + 1:
            break
        print(image_id, similarity_measure_val)
        count = count + 1
        my_utils.plot_similar_images(image_id, similarity_measure_val)
    return


# cmom euclidean similarity function
def cmom_similarity(source_img,loaded_imgs, k):

    simiarity_measure_map={}

    for image_id, load_img_vec in loaded_imgs.items():
        sum=0
        for i in range(0,1728):
            sum = sum + (source_img[i]-load_img_vec[i])**2
        ed_val=math.sqrt(sum)
        simiarity_measure_map[image_id]=ed_val


    sorted_simiarity_measure_map = collections.OrderedDict(sorted(simiarity_measure_map.items(), key=lambda measure_val: measure_val[1]))
    count=0
    print(k)
    for image_id, similarity_measure_val in sorted_simiarity_measure_map.items():
        if count == int(k)+1:
            break
        print(image_id, similarity_measure_val)
        count = count + 1
        my_utils.plot_similar_images(image_id, similarity_measure_val)
    return

# similarity function for sift using euclidean
def sift_similarity(source_img,loaded_imgs, k):

    len1 = len(source_img)
    simiarity_measure_map = {}
    for image_id, load_img_vec in loaded_imgs.items():

        s_val = my_utils.calc_sift_distance(source_img, load_img_vec)
        simiarity_measure_map[image_id] = s_val

    sorted_simiarity_measure_map = collections.OrderedDict(sorted(simiarity_measure_map.items(), key=lambda measure_val: measure_val[1]))

    count = 0

    for image_id, similarity_measure_val in sorted_simiarity_measure_map.items():
        if count == int(k) + 1:
            break
        print(image_id, similarity_measure_val)
        count = count + 1
        my_utils.plot_similar_images(image_id, similarity_measure_val)
    return




