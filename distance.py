import pandas as pd
import scipy.io as sp
from scipy.spatial import distance
from sklearn import manifold, datasets, decomposition, discriminant_analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib import offsetbox



# ################# Uncomment for MNIST###########################
# df1= pd.read_csv('mnist_train_labels.csv', names=['label'])
#
# df2= pd.read_csv('mnist_train.csv', header=None)

########for a small dataset
df1= pd.read_csv('labels_mnist_1000.csv', names=['label'])

df2= pd.read_csv('images_mnist_1000.csv', header=None)



# ###################Uncomment for CIFAR########################
# df1= pd.read_csv('cifar_train_labels.csv', names=['label'])
#
# df2= pd.read_csv('cifar_train_data.csv', header=None)


# ###################Uncomment for CIFAR########################
# df1= pd.read_csv('labels_cifar_500.csv', names=['label'])
#
# df2= pd.read_csv('images_cifar_500.csv', header=None)


X = df2.values
y = df1.values

n_samples, n_features = X.shape

##############for LDA based dimension reduction############ Read : https://pythonmachinelearning.pro/dimensionality-reduction/
# n_samples, n_features = X.shape

# def embedding_plot(X, title):
#     x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
#     X = (X - x_min) / (x_max - x_min)
#
#     plt.figure()
#     ax = plt.subplot(aspect='equal')
#     sc = ax.scatter(X[:, 0], X[:, 1], lw=0, s=40, c=y / 10.)
#
#     shown_images = np.array([[1., 1.]])
#     for i in range(X.shape[0]):
#         if np.min(np.sum((X[i] - shown_images) ** 2, axis=1)) < 1e-2: continue
#         shown_images = np.r_[shown_images, [X[i]]]
#         ax.add_artist(offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]))
#
#     plt.xticks([]), plt.yticks([])
#     plt.title(title)

########### Uncomment for LDA
# X_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)

############

# embedding_plot(X_lda, "LDA")
#
# plt.show()



#########t-SNE###################

X_lda = manifold.TSNE(n_components=2, init='pca').fit_transform(X)


#################################################################
# X_lda = PCA(n_components=2).fit_transform(X)

# print(X_lda)
combined = np.hstack((y,X_lda))

# is_one =  combined[]
#
# one = combined[is_one]

all_digit_df = pd.DataFrame(combined)
# result = pd.concat([df1, df2], axis=1, sort=False)
#


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




########################## make new df without labels #############


zero_df_0 = zero_df.iloc[:,1:3]
zero_df_1 = one_df.iloc[:,1:3]
zero_df_2 = two_df.iloc[:,1:3]
zero_df_3 = three_df.iloc[:,1:3]
zero_df_4 = four_df.iloc[:,1:3]
zero_df_5 = five_df.iloc[:,1:3]
zero_df_6 = six_df.iloc[:,1:3]
zero_df_7 = seven_df.iloc[:,1:3]
zero_df_8 = eight_df.iloc[:,1:3]
zero_df_9 = nine_df.iloc[:,1:3]




######################## convert each digits to numpy for processing ######

zero_np = zero_df_0.values
one_np = zero_df_1.values
two_np = zero_df_2.values
three_np = zero_df_3.values
four_np = zero_df_4.values
five_np = zero_df_5.values
six_np = zero_df_6.values
seven_np = zero_df_7.values
eight_np = zero_df_8.values
nine_np = zero_df_9.values



########## find mean for one image of each class #############
mean_zero = np.mean(zero_np,axis =0)
mean_one = np.mean(one_np, axis =0)
mean_two = np.mean(two_np, axis =0)
mean_three = np.mean(three_np, axis =0)
mean_four = np.mean(four_np, axis =0)
mean_five = np.mean(five_np, axis =0)
mean_six = np.mean(six_np, axis =0)
mean_seven = np.mean(seven_np, axis =0)
mean_eight = np.mean(eight_np, axis =0)
mean_nine = np.mean(nine_np, axis =0)

print('Mean 0 of class is : ' ,mean_zero)
print('Mean 1 of class is : ' ,mean_one)
print('Mean 2 of class is : ' ,mean_two)
print('Mean 3 of class is : ' ,mean_three)
print('Mean 4 of class is : ' ,mean_four)
print('Mean 5 of class is : ' ,mean_five)
print('Mean 6 of class is : ' ,mean_six)
print('Mean 7 of class is : ' ,mean_seven)
print('Mean 8 of class is : ' ,mean_eight)
print('Mean 9 of class is : ' ,mean_nine)




mean_zero = np.array([mean_zero])
mean_one = np.array([mean_one])
mean_two = np.array([mean_two])
mean_three = np.array([mean_three])
mean_four = np.array([mean_four])
mean_five = np.array([mean_five])
mean_six = np.array([mean_six])
mean_seven = np.array([mean_seven])
mean_eight = np.array([mean_eight])
mean_nine = np.array([mean_nine])

######### Uncomment to find distance between different labels ###########
new = np.vstack((mean_zero,mean_one,mean_two,mean_three,mean_four,mean_five,mean_six,mean_seven,mean_eight,mean_nine))

dst = distance.pdist(new,'euclidean')

mean_dst = np.min(dst)
#
# ############# Uncomment this for t-SNE and cdist ###########
#
#
#
# ###distances between label 0 and others
# dst1_label0 = distance.cdist(mean_zero,mean_one,'euclidean')
# dst2_label0 = distance.cdist(mean_zero,mean_two,'euclidean')
# dst3_label0 = distance.cdist(mean_zero,mean_three,'euclidean')
# dst4_label0 = distance.cdist(mean_zero,mean_four,'euclidean')
# dst5_label0 = distance.cdist(mean_zero,mean_five,'euclidean')
# dst6_label0 = distance.cdist(mean_zero,mean_six,'euclidean')
# dst7_label0 = distance.cdist(mean_zero,mean_seven,'euclidean')
# dst8_label0 = distance.cdist(mean_zero,mean_eight,'euclidean')
# dst9_label0 = distance.cdist(mean_zero,mean_nine,'euclidean')
#
# total_array1 = np.array([dst1_label0,dst2_label0,dst3_label0,dst4_label0, dst5_label0, dst6_label0, dst7_label0, dst8_label0, dst9_label0])
#
# # total = np.sum(dst1_label0,dst2_label0,dst3_label0,dst4_label0,dst5_label0,dst6_label0,dst7_label0,dst8_label0,dst9_label0)
# total1 = np.sum(total_array1)
# # print(dst9_label0)
#
# ###### scaled scores
#
#
# # print(total1)
#
# ###############################
# ###distances between label 1 and others
# dst1_label1 = distance.cdist(mean_one,mean_two,'euclidean')
# dst2_label1 = distance.cdist(mean_one,mean_three,'euclidean')
# dst3_label1 = distance.cdist(mean_one,mean_four,'euclidean')
# dst4_label1 = distance.cdist(mean_one,mean_five,'euclidean')
# dst5_label1 = distance.cdist(mean_one,mean_six,'euclidean')
# dst6_label1 = distance.cdist(mean_one,mean_seven,'euclidean')
# dst7_label1 = distance.cdist(mean_one,mean_eight,'euclidean')
# dst8_label1 = distance.cdist(mean_one,mean_nine,'euclidean')
#
# total_array2 = np.array([dst1_label1,dst2_label1,dst3_label1,dst4_label1, dst5_label1, dst6_label1, dst7_label1, dst8_label1])
#
# total2 = np.sum(total_array2)
# # print(total2)
# # ###distances between label 2 and others#######
# dst1_label2 = distance.cdist(mean_two,mean_three,'euclidean')
# dst2_label2 = distance.cdist(mean_two,mean_four,'euclidean')
# dst3_label2 = distance.cdist(mean_two,mean_five,'euclidean')
# dst4_label2 = distance.cdist(mean_two,mean_six,'euclidean')
# dst5_label2 = distance.cdist(mean_two,mean_seven,'euclidean')
# dst6_label2 = distance.cdist(mean_two,mean_eight,'euclidean')
# dst7_label2 = distance.cdist(mean_two,mean_nine,'euclidean')
#
# total_array3 = np.array([dst1_label2,dst2_label2,dst3_label2,dst4_label2, dst5_label2, dst6_label2, dst7_label2])
# total3 = np.sum(total_array3)
#
# ###distances between label 3 and others#######
# dst1_label3 = distance.cdist(mean_three,mean_four,'euclidean')
# dst2_label3 = distance.cdist(mean_three,mean_five,'euclidean')
# dst3_label3 = distance.cdist(mean_three,mean_six,'euclidean')
# dst4_label3 = distance.cdist(mean_three,mean_seven,'euclidean')
# dst5_label3 = distance.cdist(mean_three,mean_eight,'euclidean')
# dst6_label3 = distance.cdist(mean_three,mean_nine,'euclidean')
#
# total_array4 = np.array([dst1_label3,dst2_label3,dst3_label3,dst4_label3, dst5_label3, dst6_label3])
# total4 = np.sum(total_array4)
#
#
# ###distances between label 4 and others#######
# dst1_label4 = distance.cdist(mean_four,mean_five,'euclidean')
# dst2_label4 = distance.cdist(mean_four,mean_six,'euclidean')
# dst3_label4 = distance.cdist(mean_four,mean_seven,'euclidean')
# dst4_label4 = distance.cdist(mean_four,mean_eight,'euclidean')
# dst5_label4 = distance.cdist(mean_four,mean_nine,'euclidean')
#
# total_array5 = np.array([dst1_label4,dst2_label4,dst3_label4,dst4_label4, dst5_label4])
# total5 = np.sum(total_array5)
#
#
# ###distances between label 5 and others#######
# dst1_label5 = distance.cdist(mean_five,mean_six,'euclidean')
# dst2_label5 = distance.cdist(mean_five,mean_seven,'euclidean')
# dst3_label5 = distance.cdist(mean_five,mean_eight,'euclidean')
# dst4_label5 = distance.cdist(mean_five,mean_nine,'euclidean')
#
# total_array6 = np.array([dst1_label5,dst2_label5,dst3_label5,dst4_label5])
# total6 = np.sum(total_array6)
#
# ###distances between label 6 and others#######
# dst1_label6 = distance.cdist(mean_six,mean_seven,'euclidean')
# dst2_label6 = distance.cdist(mean_six,mean_eight,'euclidean')
# dst3_label6 = distance.cdist(mean_six,mean_nine,'euclidean')
#
# total_array7 = np.array([dst1_label6,dst2_label6,dst3_label6])
# total7 = np.sum(total_array7)
#
#
# ###distances between label 7 and others#######
# dst1_label7 = distance.cdist(mean_seven,mean_eight,'euclidean')
# dst2_label7 = distance.cdist(mean_seven,mean_nine,'euclidean')
#
# total_array8 = np.array([dst1_label7,dst2_label7])
# total8 = np.sum(total_array8)
#
# # ###distances between label 8 and others#######
# dst1_label8 = distance.cdist(mean_eight,mean_nine,'euclidean')
# total9 = np.array([dst1_label8])
#
#
# total_array = np.array([total1,total2,total3,total4,total5,total6,total7,total8])
# total = np.sum(total_array)
#
# # print(total)
#
#
# #################### Find scaled scores ##################
#
#
# #### for label 0
#
# scaled_label0_1 = dst1_label0/ total
# scaled_label0_2 = dst2_label0/ total
# scaled_label0_3 = dst3_label0/ total
# scaled_label0_4 = dst4_label0/ total
# scaled_label0_5 = dst5_label0/ total
# scaled_label0_6 = dst6_label0/ total
# scaled_label0_7 = dst7_label0/ total
# scaled_label0_8 = dst8_label0/ total
# scaled_label0_9 = dst9_label0/ total
#
#
# scaled_label0 = np.array([scaled_label0_1,scaled_label0_2,scaled_label0_3,scaled_label0_4,scaled_label0_5,scaled_label0_6,scaled_label0_7,scaled_label0_8,scaled_label0_9])
#
# # print(scaled_label0)
# ###for label 1
# scaled_label1_1 = dst1_label1/ total
# scaled_label1_2 = dst2_label1/ total
# scaled_label1_3 = dst3_label1/ total
# scaled_label1_4 = dst4_label1/ total
# scaled_label1_5 = dst5_label1/ total
# scaled_label1_6 = dst6_label1/ total
# scaled_label1_7 = dst7_label1/ total
# scaled_label1_8 = dst8_label1/ total
#
#
# scaled_label1 = np.array([scaled_label1_1,scaled_label1_2,scaled_label1_3,scaled_label1_4,scaled_label1_5,scaled_label1_6,scaled_label1_7,scaled_label1_8])
#
# ####for label 2
#
# scaled_label2_1 = dst1_label2/ total
# scaled_label2_2 = dst2_label2/ total
# scaled_label2_3 = dst3_label2/ total
# scaled_label2_4 = dst4_label2/ total
# scaled_label2_5 = dst5_label2/ total
# scaled_label2_6 = dst6_label2/ total
# scaled_label2_7 = dst7_label2/ total
#
# scaled_label2 = np.array([scaled_label2_1,scaled_label2_2,scaled_label2_3,scaled_label2_4,scaled_label2_5,scaled_label2_6,scaled_label2_7])
#
# #### for label 3
# scaled_label3_1 = dst1_label3/ total
# scaled_label3_2 = dst2_label3/ total
# scaled_label3_3 = dst3_label3/ total
# scaled_label3_4 = dst4_label3/ total
# scaled_label3_5 = dst5_label3/ total
# scaled_label3_6 = dst6_label3/ total
#
# scaled_label3 = np.array([scaled_label3_1,scaled_label3_2,scaled_label3_3,scaled_label3_4,scaled_label3_5,scaled_label3_6])
#
# ####for label 4
# scaled_label4_1 = dst1_label4/ total
# scaled_label4_2 = dst2_label4/ total
# scaled_label4_3 = dst3_label4/ total
# scaled_label4_4 = dst4_label4/ total
# scaled_label4_5 = dst5_label4/ total
#
# scaled_label4 = np.array([scaled_label4_1,scaled_label4_2,scaled_label4_3,scaled_label4_4,scaled_label4_5])
#
# ###for label 5
#
# scaled_label5_1 = dst1_label5/ total
# scaled_label5_2 = dst2_label5/ total
# scaled_label5_3 = dst3_label5/ total
# scaled_label5_4 = dst4_label5/ total
#
# scaled_label5 = np.array([scaled_label5_1,scaled_label5_2,scaled_label5_3,scaled_label5_4])
#
# ###for label 6
# scaled_label6_1 = dst1_label6/ total
# scaled_label6_2 = dst2_label6/ total
# scaled_label6_3 = dst3_label6/ total
#
# scaled_label6 = np.array([scaled_label6_1,scaled_label6_2,scaled_label6_3])
#
# ##for label 7
#
# scaled_label7_1 = dst1_label7/ total
# scaled_label7_2 = dst2_label7/ total
#
# scaled_label7 = np.array([scaled_label7_1,scaled_label7_2])
# ### for label 8
# scaled_label8_1 = dst1_label8/ total
#
# scaled_label8 = np.array([scaled_label8_1])
#
# scaled = np.concatenate([scaled_label0,scaled_label1,scaled_label2,scaled_label3,scaled_label4,scaled_label5,scaled_label6,scaled_label7,scaled_label8])
#
#
# # print(scaled)
#
# scaled_score = np.mean(scaled)
#
# print('Control_score for the dataset is: ',scaled_score )


# ################### Uncomment to find distance within label first ##############
#
# dst_0 = distance.pdist(zero_np,'euclidean')
# mean_dst_0 = np.mean(dst_0)
#
# dst_1 = distance.pdist(one_np,'euclidean')
# mean_dst_1 = np.mean(dst_1)
#
# dst_2 = distance.pdist(two_np,'euclidean')
# mean_dst_2 = np.mean(dst_2)
#
# dst_3 = distance.pdist(three_np,'euclidean')
# mean_dst_3 = np.mean(dst_3)
#
# dst_4 = distance.pdist(four_np,'euclidean')
# mean_dst_4 = np.mean(dst_4)
#
# dst_5 = distance.pdist(five_np,'euclidean')
# mean_dst_5 = np.mean(dst_5)
#
# dst_6 = distance.pdist(six_np,'euclidean')
# mean_dst_6 = np.mean(dst_6)
#
# dst_7 = distance.pdist(seven_np,'euclidean')
# mean_dst_7 = np.mean(dst_7)
#
# dst_8 = distance.pdist(eight_np,'euclidean')
# mean_dst_8 = np.mean(dst_8)
#
# dst_9 = distance.pdist(zero_np,'euclidean')
# mean_dst_9 = np.mean(dst_9)
#
# mean_dst_np = np.vstack((mean_dst_0,mean_dst_1,mean_dst_2,mean_dst_3,mean_dst_4,mean_dst_5,mean_dst_6,mean_dst_7,mean_dst_8,mean_dst_9))

# mean_dst = np.mean(mean_dst_np)








################ Print the mean distance among various labels##############

print('Mean Distance is : ',mean_dst)











########################## Old approach ############## To be deldeted later
# dst0_1=  distance.euclidean(mean_zero,mean_one)
# dst0_2=  distance.euclidean(mean_zero,mean_two)
# dst0_3=  distance.euclidean(mean_zero,mean_three)
# dst0_4=  distance.euclidean(mean_zero,mean_four)
# dst0_5=  distance.euclidean(mean_zero,mean_five)
# dst0_6=  distance.euclidean(mean_zero,mean_six)
# dst0_7=  distance.euclidean(mean_zero,mean_seven)
# dst0_8=  distance.euclidean(mean_zero,mean_eight)
# dst0_9=  distance.euclidean(mean_zero,mean_nine)
# dst1_2=  distance.euclidean(mean_one,mean_two)
# dst1_3=  distance.euclidean(mean_one,mean_two)
# dst1_4=  distance.euclidean(mean_one,mean_two)
# dst1_5=  distance.euclidean(mean_two,mean_two)
# dst1_6=  distance.euclidean(mean_three,mean_two)
# dst1_7=  distance.euclidean(mean_zero,mean_two)
# dst1_8=  distance.euclidean(mean_zero,mean_two)
# dst1_9=  distance.euclidean(mean_zero,mean_two)
# dst2_3=  distance.euclidean(mean_zero,mean_two)
# dst2_4=  distance.euclidean(mean_zero,mean_two)
# dst2_5=  distance.euclidean(mean_zero,mean_two)
# dst2_6=  distance.euclidean(mean_zero,mean_two)
# dst2_7=  distance.euclidean(mean_zero,mean_two)
# dst2_8=  distance.euclidean(mean_zero,mean_two)
# dst2_9=  distance.euclidean(mean_zero,mean_two)

# zero_single = zero.iloc[0]
# one_single = one.iloc[0]
# two_single = two.iloc[0]
# three_single = three.iloc[0]
# four_single = four.iloc[0]
# five_single = five.iloc[0]
# six_single = six.iloc[0]
# seven_single = seven.iloc[0]
# eight_single = eight.iloc[0]
# nine_single = nine.iloc[0]
#
# one_image_per_class = pd.concat([zero_single, one_single,two_single,three_single,four_single,five_single,six_single,seven_single,eight_single,nine_single], axis=1, sort=False)
#
#
# reduced_data = PCA(n_components=2).fit_transform(*zero_single)
# # distance_zero_to_one = distance.pdist(one_image_per_class,'euclidean')
#
#
