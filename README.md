## Improving prostate cancer diagnostic pathway with a multi-parametric generative self-supervised approach.
### Description:
Prostate cancer (PCa) is the fifth leading cause of death worldwide. However, diagnosis of PCa based on MRI is challenging because of the time it takes to analyze the images and the variability between the readers. Moreover, an accurate segmentation of the prostate is of vital relevance due to the fact that the diagnostic depends on an accurate characterization of it (volume of the prostate, for instance). Furthermore, treatment for PCa depends on the accuracy of the segmentation too. Convolutional neural networks (CNNs) have been the de facto standard for nowadays 3D and 2D medical image classification and segmentation. However, CNNs suffer when the amount of data is scarce, being hard to train and not reaching optimal solutions when trained with a limited amount of data. Self-supervised learning techniques aim to train CNNs in such a way that they’re able to learn even in limited data regimes.

In this work, we will explore a self-supervised approach to deal with a limited amount of magnetic resonance images (MRI) of the prostate. We will implement a self-supervised approach able to integrate bi-parametric MRI (ADC and T2w). We will test our approach in a limited amount of data regime and for two relevant applications in PCa: segmentation and classification between abnormal (MRI with tumors) and control images (without tumors).

convolutional neural networks (CNNs), autoencoders, self-supervised learning
