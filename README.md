# deep-loki
Advances in Deep Learning have lead to the creation of learning agents that can perform non-trivial human level tasks (such as visual detection, speech recognition etc.). However, there exist inherent vulnerabilities in these systems. These vulnerabilities can be methodically exploited to manipulate the output of the system by subtly changing the input. These changes are almost imperceptible to the naked eye and are termed as adversarial perturbations. In context of visual systems, perturbed images are by design meant to fool networks that are otherwise deemed to be reliable by predicting incorrect classes with confidence. 

I am currently understanding how subspaces in the feature space where these adversarial examples exist relate to the high dimensional decision boundaries and the internal representations made by the model. We are particularly interested in the applications of this field to **Deepfakes** , which has started to make headlines as a political and social tool to spread misinformation. We are attempting to generate *natural* adversarial examples that modify existing artifacts of the image rather
than add noise that can be potentially defended against or be detected.

## Benchmarks

### Fast Gradient Sign Method
ResNet-50 Accuracy  |DenseNet-121 Accuracy |Average Confidence | Attack Parameters |
-----------------|-----------------|-----------------|-----------------|
13.54%  | 19.02%  | - | Epsilon is similar to 0.0005 |

![FGSM_example](https://github.com/divyam02/deep-loki/blob/master/images/FGSM_1.png)

### Jacobian Saliency Map Attack
ResNet-50 Accuracy  |DenseNet-121 Accuracy  |Average Confidence | Attack Parameters |
-----------------|-----------------|-----------------|-----------------|
12.49%  | 11.54%  |About 90%  | L1 norm of the perturbed image is restricted to 50. Tested around 1000 images evenly distributed amongst classes. |

![JSMA_example](https://github.com/divyam02/deep-loki/blob/master/images/JSMA_example.jpg)

### Deep Fool
ResNet-50 Accuracy  |DenseNet-121 Accuracy |Average Confidence | Attack Parameters |
-----------------|-----------------|-----------------|-----------------|
10.39%  | 10.87%  | - | Overshoot: 0.002, Max iters: 50. See implementational details.|

![DF_example](https://github.com/divyam02/deep-loki/blob/master/images/DF_example.jpg)

### Carlini-Wagner Attack
ResNet-50 Accuracy  |DenseNet-121 Accuracy |Average Confidence | Attack Parameters |
-----------------|-----------------|-----------------|-----------------|
0.0%  | 0.0%  | About 97% | L2 norm values vary. Tested 500 images evenly distributed from all classes. |

![CWl2_example](https://github.com/divyam02/deep-loki/blob/master/images/CWl2_example.jpg)

### Universal Perturbation Perturbation
ResNet-50 Accuracy  |DenseNet-121 Accuracy |Average Confidence | Attack Parameters |
-----------------|-----------------|-----------------|-----------------|
14.11%  | 14.19%  | 92.12% | l∞ norm is limited to 0.5. Deep Fool max iters to find intermediate perturbation: 10. Tested after obtaining perturbation on training on 10000 images from the training set. |

![UAP_example](https://github.com/divyam02/deep-loki/blob/master/images/UAP_example.jpg)

### One Pixel Attack
ResNet-50 Accuracy  |DenseNet-121 Accuracy |Average Confidence | Attack Parameters |
-----------------|-----------------|-----------------|-----------------|
40.83%  | -  | Reportedly 97.74% | Differential Evolution iterations: 150. Tested for 200 images evenly distributed from each class due to slow processing. |

![OHKO_example](https://github.com/divyam02/deep-loki/blob/master/images/OHKO_example.jpg)

### Most Significant Bit Attack
ResNet-50 Accuracy  |DenseNet-121 Accuracy |Average Confidence | Attack Parameters |
-----------------|-----------------|-----------------|-----------------|
28.10%  | 28.77%  | - | 12% pixels were perturbed |

![MSB_example](https://github.com/divyam02/deep-loki/blob/master/images/MSB_example.jpg)



 ## Shift in Attention of Perturbed Images
To visualize how the attention of a classifier shifts with an adversarial perturbation. I used an implementation of Grad-CAM with the Carlini & Wagner attack, JSMA and the One Pixel attack to obtain Heat Maps (indicating the important regions the classifier looks for in the image), Guided Backpropagation (highlights all contributing features for classification) and Grad-CAM (highlights class discriminative features) mappings. The intensity of pixels indicates prominence of features compared to others.

![](https://github.com/divyam02/deep-loki/blob/master/images/CWL2_heat_map_1.png)
![](https://github.com/divyam02/deep-loki/blob/master/images/CWL2_heat_map_2.png)

(Left to Right) Targeted attacks on two classes, Feature importance heatmaps, Guided backprop and guided backprop with Grad-CAM, using C&W attack. The original image is to the top and the adversary is below.

![](https://github.com/divyam02/deep-loki/blob/master/images/JSMA_heat_map_1.png)
![](https://github.com/divyam02/deep-loki/blob/master/images/JSMA_example_2.png)

(Left to Right) Targeted attacks on two classes, Feature importance heatmaps, Guided backprop andguided backprop Grad-CAM, using JSMA. The original image is to the top and the adversary is below. 

![](https://github.com/divyam02/deep-loki/blob/master/images/one_pixel_example_1.png)
![](https://github.com/divyam02/deep-loki/blob/master/images/one_pixel_example_2.png)

(Left to Right) Targeted attacks on two classes, Feature importance heatmaps, Guided backprop and guided backprop Grad-CAM, using One Pixel Attack. The original image is to the top and the adversary is below Note that while these images are perturbed, they are not misclassified as are the images in the examples above.

### Acknowledgements
Part of an ongoing project under Dr. Mayank Vatsa and Dr. Richa Singh
