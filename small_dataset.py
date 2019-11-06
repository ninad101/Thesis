import pandas as pd
import scipy.io as sp
from scipy.spatial import distance
from sklearn import manifold, datasets, decomposition, discriminant_analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox




# df1= pd.read_csv('mnist_train_labels.csv', names=['label'])
#
# df2= pd.read_csv('mnist_train.csv', header=None)



df1= pd.read_csv('cifar_train_labels.csv', names=['label'])

df2= pd.read_csv('cifar_train_data.csv', header=None)

X = df2.values
y = df1.values


combined = np.hstack((y,X))


all_digit_df = pd.DataFrame(combined)

################## separate all labels and corresponding data using dataframes ########
zero_df = all_digit_df.loc[df1.label ==0]
one_df = all_digit_df.loc[df1.label ==1]
two_df = all_digit_df.loc[df1.label ==2]
three_df = all_digit_df.loc[df1.label ==3]
four_df = all_digit_df.loc[df1.label ==4]
five_df  = all_digit_df.loc[df1.label ==5]
six_df = all_digit_df.loc[df1.label ==6]
seven_df  = all_digit_df.loc[df1.label ==7]
eight_df = all_digit_df.loc[df1.label ==8]
nine_df = all_digit_df.loc[df1.label ==9]


#################### Only labels dataframe ############################


label_zero = zero_df.iloc[:,0]
label_one = one_df.iloc[:,0]
label_two = two_df.iloc[:,0]
label_three = three_df.iloc[:,0]
label_four = four_df.iloc[:,0]
label_five = five_df.iloc[:,0]
label_six = six_df.iloc[:,0]
label_seven = seven_df.iloc[:,0]
label_eight = eight_df.iloc[:,0]
label_nine = nine_df.iloc[:,0]


# ########################## make new df without labels for MNIST #############
# zero_df_0 = zero_df.iloc[:,1:785]
# zero_df_1 = one_df.iloc[:,1:785]
# zero_df_2 = two_df.iloc[:,1:785]
# zero_df_3 = three_df.iloc[:,1:785]
# zero_df_4 = four_df.iloc[:,1:785]
# zero_df_5 = five_df.iloc[:,1:785]
# zero_df_6 = six_df.iloc[:,1:785]
# zero_df_7 = seven_df.iloc[:,1:785]
# zero_df_8 = eight_df.iloc[:,1:785]
# zero_df_9 = nine_df.iloc[:,1:785]

########################## make new df without labels for CIFAR #############
zero_df_0 = zero_df.iloc[:,1:1025]
zero_df_1 = one_df.iloc[:,1:1025]
zero_df_2 = two_df.iloc[:,1:1025]
zero_df_3 = three_df.iloc[:,1:1025]
zero_df_4 = four_df.iloc[:,1:1025]
zero_df_5 = five_df.iloc[:,1:1025]
zero_df_6 = six_df.iloc[:,1:1025]
zero_df_7 = seven_df.iloc[:,1:1025]
zero_df_8 = eight_df.iloc[:,1:1025]
zero_df_9 = nine_df.iloc[:,1:1025]

#####################################################################
######################### Make Smaller Datasets #####################
#####################################################################



# # ########################## Uncomment for 1000 Images per Class ######################
#
#
# ######################### get first 1000 labels for all classes, combine and save in csv ##############################
# labels_zero_1000 = label_zero.head(1000)
# labels_one_1000 = label_one.head(1000)
# labels_two_1000 = label_two.head(1000)
# labels_three_1000 = label_three.head(1000)
# labels_four_1000 = label_four.head(1000)
# labels_five_1000 = label_five.head(1000)
# labels_six_1000 = label_six.head(1000)
# labels_seven_1000 = label_seven.head(1000)
# labels_eight_1000 = label_eight.head(1000)
# labels_nine_1000 = label_nine.head(1000)
#
# labels = pd.concat([labels_zero_1000,labels_one_1000,labels_two_1000,labels_three_1000,labels_four_1000,labels_five_1000,labels_six_1000,labels_seven_1000,labels_eight_1000,labels_nine_1000], axis=0)
# labels_1000 = labels.reset_index(drop=True)
#
# labels_1000.to_csv ('labels_cifar_1000.csv', index = None, header=True)
#
# ###################### get first 1000 Images for all classes, combine and save in csv ################################
#
# zero_1000 = zero_df_0.head(1000)
# one_1000 = zero_df_1.head(1000)
# two_1000 = zero_df_2.head(1000)
# three_1000 = zero_df_3.head(1000)
# four_1000 = zero_df_4.head(1000)
# five_1000 = zero_df_5.head(1000)
# six_1000 = zero_df_6.head(1000)
# seven_1000 = zero_df_7.head(1000)
# eight_1000 = zero_df_8.head(1000)
# nine_1000 = zero_df_9.head(1000)
#
# images = pd.concat([zero_1000,one_1000,two_1000,three_1000,four_1000,five_1000,six_1000,seven_1000,eight_1000,nine_1000], axis=0)
# images_1000 = images.reset_index(drop=True)
#
# print(images_1000)
#
# images_1000.to_csv ('images_cifar_1000.csv', index = None, header=True)
#
#
# # ########################################################################################################################
#
#
# ########################## 500 Images per Class ######################
#
#
# ######################### get first 500 labels for all classes, combine and save in csv ##############################
# labels_zero_500 = label_zero.head(500)
# labels_one_500 = label_one.head(500)
# labels_two_500 = label_two.head(500)
# labels_three_500 = label_three.head(500)
# labels_four_500 = label_four.head(500)
# labels_five_500 = label_five.head(500)
# labels_six_500 = label_six.head(500)
# labels_seven_500 = label_seven.head(500)
# labels_eight_500 = label_eight.head(500)
# labels_nine_500 = label_nine.head(500)
#
# labels = pd.concat([labels_zero_500,labels_one_500,labels_two_500,labels_three_500,labels_four_500,labels_five_500,labels_six_500,labels_seven_500,labels_eight_500,labels_nine_500], axis=0)
# labels_500 = labels.reset_index(drop=True)
#
# labels_500.to_csv ('labels_cifar_500.csv', index = None, header=True)
#
# ###################### get first 500 Images for all classes, combine and save in csv ################################
#
# zero_500 = zero_df_0.head(500)
# one_500 = zero_df_1.head(500)
# two_500 = zero_df_2.head(500)
# three_500 = zero_df_3.head(500)
# four_500 = zero_df_4.head(500)
# five_500 = zero_df_5.head(500)
# six_500 = zero_df_6.head(500)
# seven_500 = zero_df_7.head(500)
# eight_500 = zero_df_8.head(500)
# nine_500 = zero_df_9.head(500)
#
# images = pd.concat([zero_500,one_500,two_500,three_500,four_500,five_500,six_500,seven_500,eight_500,nine_500], axis=0)
# images_500 = images.reset_index(drop=True)
#
# print(images_500)
#
# images_500.to_csv ('images_cifar_500.csv', index = None, header=True)
#
#
# ########################################################################################################################
#
#
# ########################################################################################################################



# ########################## 100 Images per Class ######################
#
#
# ######################### get first 100 labels for all classes, combine and save in csv ##############################
# labels_zero_100 = label_zero.head(100)
# labels_one_100 = label_one.head(100)
# labels_two_100 = label_two.head(100)
# labels_three_100 = label_three.head(100)
# labels_four_100 = label_four.head(100)
# labels_five_100 = label_five.head(100)
# labels_six_100 = label_six.head(100)
# labels_seven_100 = label_seven.head(100)
# labels_eight_100 = label_eight.head(100)
# labels_nine_100 = label_nine.head(100)
#
# labels = pd.concat([labels_zero_100,labels_one_100,labels_two_100,labels_three_100,labels_four_100,labels_five_100,labels_six_100,labels_seven_100,labels_eight_100,labels_nine_100], axis=0)
# labels_100 = labels.reset_index(drop=True)
#
# labels_100.to_csv ('labels_cifar_100.csv', index = None, header=True)
#
# ###################### get first 100 Images for all classes, combine and save in csv ################################
#
# zero_100 = zero_df_0.head(100)
# one_100 = zero_df_1.head(100)
# two_100 = zero_df_2.head(100)
# three_100 = zero_df_3.head(100)
# four_100 = zero_df_4.head(100)
# five_100 = zero_df_5.head(100)
# six_100 = zero_df_6.head(100)
# seven_100 = zero_df_7.head(100)
# eight_100 = zero_df_8.head(100)
# nine_100 = zero_df_9.head(100)
#
# images = pd.concat([zero_100,one_100,two_100,three_100,four_100,five_100,six_100,seven_100,eight_100,nine_100], axis=0)
# images_100 = images.reset_index(drop=True)
#
# print(images_100)
#
# images_100.to_csv ('images_cifar_100.csv', index = None, header=True)
#
#
# ########################################################################################################################
#
# # ########################## 50 Images per Class ######################
#
#
# ######################### get first 50 labels for all classes, combine and save in csv ##############################
# labels_zero_50 = label_zero.head(50)
# labels_one_50 = label_one.head(50)
# labels_two_50 = label_two.head(50)
# labels_three_50 = label_three.head(50)
# labels_four_50 = label_four.head(50)
# labels_five_50 = label_five.head(50)
# labels_six_50 = label_six.head(50)
# labels_seven_50 = label_seven.head(50)
# labels_eight_50 = label_eight.head(50)
# labels_nine_50 = label_nine.head(50)
#
# labels = pd.concat([labels_zero_50,labels_one_50,labels_two_50,labels_three_50,labels_four_50,labels_five_50,labels_six_50,labels_seven_50,labels_eight_50,labels_nine_50], axis=0)
# labels_50 = labels.reset_index(drop=True)
#
# labels_50.to_csv ('labels_cifar_50.csv', index = None, header=True)
#
# ###################### get first 50 Images for all classes, combine and save in csv ################################
#
# zero_50 = zero_df_0.head(50)
# one_50 = zero_df_1.head(50)
# two_50 = zero_df_2.head(50)
# three_50 = zero_df_3.head(50)
# four_50 = zero_df_4.head(50)
# five_50 = zero_df_5.head(50)
# six_50 = zero_df_6.head(50)
# seven_50 = zero_df_7.head(50)
# eight_50 = zero_df_8.head(50)
# nine_50 = zero_df_9.head(50)
#
# images = pd.concat([zero_50,one_50,two_50,three_50,four_50,five_50,six_50,seven_50,eight_50,nine_50], axis=0)
# images_50 = images.reset_index(drop=True)
#
# print(images_50)
#
# images_50.to_csv ('images_cifar_50.csv', index = None, header=True)
#
#
# ########################################################################################################################
#
#
#
# ########################## 10 Images per Class ######################
#
#
# ######################### get first 10 labels for all classes, combine and save in csv ##############################
# labels_zero_10 = label_zero.head(10)
# labels_one_10 = label_one.head(10)
# labels_two_10 = label_two.head(10)
# labels_three_10 = label_three.head(10)
# labels_four_10 = label_four.head(10)
# labels_five_10 = label_five.head(10)
# labels_six_10 = label_six.head(10)
# labels_seven_10 = label_seven.head(10)
# labels_eight_10 = label_eight.head(10)
# labels_nine_10 = label_nine.head(10)
#
# labels = pd.concat([labels_zero_10,labels_one_10,labels_two_10,labels_three_10,labels_four_10,labels_five_10,labels_six_10,labels_seven_10,labels_eight_10,labels_nine_10], axis=0)
# labels_10 = labels.reset_index(drop=True)
#
# labels_10.to_csv ('labels_cifar_10.csv', index = None, header=True)
#
# ###################### get first 10 Images for all classes, combine and save in csv ################################
#
# zero_10 = zero_df_0.head(10)
# one_10 = zero_df_1.head(10)
# two_10 = zero_df_2.head(10)
# three_10 = zero_df_3.head(10)
# four_10 = zero_df_4.head(10)
# five_10 = zero_df_5.head(10)
# six_10 = zero_df_6.head(10)
# seven_10 = zero_df_7.head(10)
# eight_10 = zero_df_8.head(10)
# nine_10 = zero_df_9.head(10)
#
# images = pd.concat([zero_10,one_10,two_10,three_10,four_10,five_10,six_10,seven_10,eight_10,nine_10], axis=0)
# images_10 = images.reset_index(drop=True)
#
# print(images_10)
#
# images_10.to_csv ('images_cifar_10.csv', index = None, header=True)
#
#
#
#
#
#
# ########################################################################################################################
#

###################### get first 1 Images for all classes, combine and save in csv ################################
# ########################## 1 Images per Class ######################
#
#
# ######################### get first 1 labels for all classes, combine and save in csv ##############################
# labels_zero_1 = label_zero.head(1)
# labels_one_1 = label_one.head(1)
# labels_two_1 = label_two.head(1)
# labels_three_1 = label_three.head(1)
# labels_four_1 = label_four.head(1)
# labels_five_1 = label_five.head(1)
# labels_six_1 = label_six.head(1)
# labels_seven_1 = label_seven.head(1)
# labels_eight_1 = label_eight.head(1)
# labels_nine_1 = label_nine.head(1)
#
# labels = pd.concat([labels_zero_1,labels_one_1,labels_two_1,labels_three_1,labels_four_1,labels_five_1,labels_six_1,labels_seven_1,labels_eight_1,labels_nine_1], axis=0)
# labels_1 = labels.reset_index(drop=True)
#
# labels_1.to_csv ('labels_cifar_1.csv', index = None, header=True)
#
# zero_1 = zero_df_0.head(1)
# one_1 = zero_df_1.head(1)
# two_1 = zero_df_2.head(1)
# three_1 = zero_df_3.head(1)
# four_1 = zero_df_4.head(1)
# five_1 = zero_df_5.head(1)
# six_1 = zero_df_6.head(1)
# seven_1 = zero_df_7.head(1)
# eight_1 = zero_df_8.head(1)
# nine_1 = zero_df_9.head(1)
#
# images = pd.concat([zero_1,one_1,two_1,three_1,four_1,five_1,six_1,seven_1,eight_1,nine_1], axis=0)
# images_1 = images.reset_index(drop=True)
#
# print(images_1)
#
# images_1.to_csv ('images_cifar_1.csv', index = None, header=True)
#
#
# ########################################################################################################################
