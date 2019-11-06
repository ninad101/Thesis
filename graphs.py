import matplotlib.pyplot as plt
import numpy as np





######################################################################
############################ N-shot MNIST#########################
#####################################################################
accuracy_new = [37.53, 71.68, 79.24, 85.43, 87.11, 89.06]
accuracy_old = [27.55, 63.72, 73.66, 82.54, 86.06, 88.16]
increase = [9.98, 7.96, 5.59, 2.89, 1.05, 0.9]


################ plot to show variation of old v/s new#################

images_class = [1, 10, 50, 100, 500, 1000]
# fig, ax = plt.subplots()



# plt.plot(images_class, accuracy_old)
# plt.plot(images_class, accuracy_new, color='red')
# # for i, txt in enumerate(images_class):
# #     ax.annotate(txt, (images_class[i], accuracy_old[i]))
# # for i, txt in enumerate(images_class):
# #     ax.annotate(txt, (images_class[i], accuracy_new[i]))
# plt.xlabel('No. of Images per Class')
# plt.ylabel('Classification Accuracy in %')
# plt.title('Variation of N-images per class with modified CD and original CD')
# plt.show()




# ##########for bar graph ##########
# N = 6
# ind = np.arange(N)
# width = 0.27       # the width of the bars
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# # yvals = [4, 9, 2]
# rects1 = ax.bar(ind, accuracy_new, width, color='r')
# # zvals = [1,2,3]
# rects2 = ax.bar(ind+width, accuracy_old, width, color='g')
# # kvals = [11,12,13]
# # rects3 = ax.bar(ind+width*2, kvals, width, color='b')
#
# ax.set_ylabel('Classification Accuracy in %')
# ax.set_xticks(ind+width)
# ax.set_xticklabels( ('1', '10', '50','100','500','1000') )
# ax.legend( (rects1[0], rects2[0]), ('New CD', 'Standard CD') )
# ax.set_xlabel('No. of Images per Class')
#
# def autolabel(rects):
#     for rect in rects:
#         h = rect.get_height()
#         ax.text(rect.get_x()+rect.get_width()/2., 1*h, '%d'%int(h),
#                 ha='center', va='bottom')
# autolabel(rects1)
# autolabel(rects2)
# plt.show()
################# plot showing increase in estimated accuracy ###########
# images_class = ['1', '10', '50', '100', '500', '1000']
# plt.bar(images_class,increase)
# plt.xlabel('No. of Images per Class')
# plt.ylabel('Estimated Accuracy in %')
# plt.title('Increase in estimated accuracy with modified CD')
# plt.show()

# #################Comparison with same computational costs####################################
#
# images_class = ['1', '10']
# comp_accuracy = [37.53, 71.68]
# plt.bar(images_class,comp_accuracy)
# plt.xlabel('No. of Images per Class')
# plt.ylabel('Accuracy in %')
# plt.title('Accuracy with modified CD with same computational cost')
# plt.show()
###################################################################
################## N-shot CIFAR : Flexibility ######################
# ##################################################################
# images_class = ['1', '10', '50', '100', '500', '1000', 'State of Art']
# cifar_accuracy =[22.08, 26.82, 27.66, 32.04, 42.14, 53.87, 65]
#
# plt.bar(images_class,cifar_accuracy)
# plt.xlabel('No. of Images per Class')
# plt.ylabel('Classification Accuracy in %')
# plt.title('N-shot with CIFAR-10 with 1024-400-10 Configuration')
# plt.show()


# ##################################################################
# ######################## Reliability ###########################
# ###############################################################
#
# #####variation with Tau_m
#
#
#
# plt.figure()
#
# tau_m = ['4(-20%)','5','6(+20%)']
# rel1_accuracy_diff = [0.12, 1, 0.09]
# plt.subplot(321)
# plt.bar(tau_m,rel1_accuracy_diff)
# plt.xlabel('Variation')
# plt.ylabel('% change')
# plt.title('Variation of tau_m ')
#
#
# ########variation with v_thr
#
# v_thr = ['4mV(-20%)','5mV','6mV(+20%)']
# thr_diff = [0.28,1,0.31]
# plt.subplot(322)
# plt.bar(v_thr,thr_diff)
# plt.xlabel('Variation')
# plt.ylabel('% change')
# plt.title('Variation of v_thr')
# #######variation with T_ref
#
# t_ref = ['1.6mS(-20%)','2mS','2.4mS(20%)']
# t_reff_diff = [1.05,1,]
# plt.show()
#
# ############################ Compatibility ################

## for membrane capacitance
# c = ['1*10^-9', '2.16*10^-8', '35.9', '46.4','59.9','77.4' ]
# c_accuracy = [17.8, 32.20, 45.20, 51.60, 51.80, 30.60]
#
# plt.plot(c,c_accuracy)
# plt.xlabel('Membrane Capacitance')
# plt.ylabel('Classification Accuracy in %')
# plt.title('variation with C_m')
#

# #########variation with t_ref
# t_ref = ['0.1ms', '0.5ms', '1ms', '2ms','5ms','10ms' ]
# t_accuracy = [76, 70, 65.20, 55.20, 33, 15.40]
#
# plt.plot(t_ref,t_accuracy)
# plt.xlabel('Refractory Period')
# plt.ylabel('Classification Accuracy in %')
# plt.title('Variation with t_ref')

# ### variation with V_thr
# v = ['0.5mV', '1mV', '2mV', '5mV', '10mV', '50mV', '55mV']
# v_accu = [27,29,38,46,49,83,86]
#
# plt.plot(v,v_accu)
# plt.xlabel('Threshold Voltage')
# plt.ylabel('Classification Accuracy in %')
# plt.title('variation with V_thr')

#### variation with input test time

# time = ['50ms','100ms','150ms','200ms']
# accu = [91.70,92.60,93.50,93.90]
# plt.plot(time,accu)
# plt.xlabel('Input test time')
# plt.ylabel('Classification Accuracy in %')
# plt.title('Variation with input test time' )
#
# plt.show()
#


# ### uncomment fot tsne
# x_tsne_10 = [1067.63590457 ,-569.71969208,189.17199894 , 237.95201126,-315.00253811, -96.66182632,110.59725889,-337.59410263,-76.35040883  ,-400.73963318]
# y_tsne_10 = [-78.23967279, -497.14153512,-82.32712498,-566.72255417, 576.71632283,-180.98921587,456.18496826, 441.84381546,-383.33149664, 345.47139099]
#
#
# x_tsne_1 = [2777.12548828,1278.50427246 ,-502.57226562,361.71209717 ,-3241.84204102 , -2344.4753418,483.49301147,-605.38934326 ,-1392.92822266  ,-1385.73742676 ]
# y_tsne_1 = [771.66178131, -667.11804199,-817.61962891, -2201.53027344,122.36573792,-1177.44213867,733.06109619,1895.1027832, 320.28152466,-2415.75341797]
#
# x_tsne_100 = [1011.51222048,-773.39258438 ,138.88880334 ,127.80100834,-109.98509057, 58.41938751,211.65020594,-358.8348825 ,-120.91928203 ,-273.10390159]
# y_tsne_100 = [51.74918432,474.66470829,238.18169191,478.0557423 ,-586.84313412,175.88869295,-231.68338063,-547.55273979,397.47378441,-462.24133991]
#
#
# x_tsne_1000 = [1036.60289366,-855.79556821,51.83011171,49.95530317 ,-68.15629405,51.96528436,161.37358982,-246.7192365, -35.2626229, -155.06648505]
# y_tsne_1000 = [-194.76134779,-419.11440112,-271.80558309,-500.09331076,595.61796728,-161.21139074, 37.27339941,578.94004023,-219.20414369,555.31940977]
#
# ### uncomment for lda
# # x_lda = [-1.52951386,0.25799782,-1.92646979,3.43550558,-1.01459279,-0.47433387,-8.10937232,6.37528084,0.0729517,2.92784182]
# # y_lda =[-9.64785791,2.61967044,0.64384316,0.48079004,1.97629224,-0.32931963,1.96429075,-0.4454877,1.2860171,1.54824009]
#
# # x_pca = [1014.85753138,-782.39591984,167.89719562,129.01168488,-107.2538694,73.40952804,191.92838759,-319.72319874,-104.51509084,-273.364824]
# # y_pca = [52.15804832,496.15142508,242.23567654,466.53698514,-589.35834317,155.12188652,-231.02644589,-513.74333087,370.37480722,-448.97228938]
#
# labels = [0,1,2,3,4,5,6,7,8,9]
#
# ###
# fig, ax = plt.subplots()
# ax.scatter(x_tsne_10, y_tsne_10,c='g')
# ax.scatter(x_tsne_100, y_tsne_100,c='b')
# ax.scatter(x_tsne_1000, y_tsne_1000,c='r')
#
#
# # ax.legend()
# ax.legend( ( '10 images per class','100 images per class','1000 images per class') )
#
# for i, txt in enumerate(labels):
#         ax.annotate(txt, (x_tsne_1000[i], y_tsne_1000[i]))
#         ax.annotate(txt, (x_tsne_100[i], y_tsne_100[i]))
#         ax.annotate(txt, (x_tsne_10[i], y_tsne_10[i]))
#
# # plt.scatter(y,x)
# plt.xlabel('X coordinate')
# plt.ylabel('Y coordinate')
# plt.title('Plot showing all 10 classes after t-SNE(mean of each cluster shift)' )
#
#
# plt.show()


#### for variation between distance and accuracy

distance =[95,99,99.5,100,100.5,101,101.5,102,102.5,103,103.5,104,104.5,105,105.5,106,106.5,107,107.5,108,108.5,109,113]
accuracy =[65,70.23,70.39,70.57,70.61,70.64,70.87,71.14,71.31,71.49,71.68,71.9,71.8,71.61,71.53,71.45,71.11,70.8,70.68,70.57,70.49,68.87, 63.67]

plt.plot(distance,accuracy)
plt.xlabel('Distance')
plt.ylabel('Classification Accuracy in %')
plt.title('Variation of distance')

plt.show()

# ###### for variation of tau
# tau =[0.1,1,10,100,1000]
# accuracy =[60.05,66.45,67.08,67.23,67.14]
#
# plt.plot(tau,accuracy)
# plt.xlabel('Tau_m')
# plt.ylabel('Classification Accuracy in %')
# plt.title('Variation of Tau_m')
#
# plt.show()

# ###### for variation of v_thresh
# vth =[5,15,35,25,50]
# accuracy =[69.06,69.43,70.45,67.53,67.23]
#
# plt.plot(vth,accuracy)
# plt.xlabel('V_thresh(mV)')
# plt.ylabel('Classification Accuracy in %')
# plt.title('Variation of Neuron membrane threshold potential')
#
# plt.show()

