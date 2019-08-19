# Paper: Universal Adversarial Perturbations
*    **Authors**: Moosavi-Dezfooli, Fawsi, Frossard
*    Link: https://arxiv.org/abs/1610.08401

### Goal
* Show existence of a image-agnostic, universal and small perturbation vector that causes natural images to be misclassified with high probability.
* Show geometric correlations between the high dimensional decision boundaries.

### Author Observations
* Generalizes surprisingly well. Perturbations computed for a small set of data points fool new images with high probability.
* Also does so across networks.

### Adversarial Perturbations
* Let μ be a distribution of images in R<sup>d</sup>.
* Let k' be a classifier that outputs estimated label k'(x) for each x ∈ R<sup>d</sup>.
* **Aim**: to find *v* ∈ R<sup>d</sup> that fools *almost* all datapoints sampled from mu.
    * ie, **k'(x+v)!=k'(x)** for most x~mu.
    * where ***v* satisfies ||v||<sub>p</sub> <= Q** (magnitude) and **P(k'(x+v)!=k'(x)) >= 1 - *delta***, where delta is some desired fooling rate.
* Let X = {*x1, .., xm*} be a set of images sampled from Mu.
* Iterate over X to obtain a minimal perturbation ∆v<sub>i</sub> that sends x<sub>i</sub>+v+∆v<sub>i</sub> to the decision boundary of the classifier. 
    * Solve the optimization problem: ∆v<sub>i</sub> <- *argmin*||r||<sub>2</sub> s.t. k'(x+v+r)!=k'(x)
    * For the magnitude constraint, v+∆v<sub>i</sub> is projected onto a *Sphere*(0, Q) satifying S(v) = *argmin*<sub>v'</sub>||v - v'||<sub>2</sub> , where ||v'||<sub>p</sub> <= Q.
    * Update v <- S(v+∆v<sub>i</sub>)
* Stop iterating after fooling rate on validation set is below threshold.

### Properties
* Since this is greedy and dependent on the order of iteration, other universal perturbations are possible.
* Seems to generalize across architectures.
* Sample images are very small in practice.
* Fine tuning with universal perturbations did not increase robustness to perturbations (much. See Deep Fool)
### Questions
* Dependence on number of target labels?
* Is every pixel of the image being modified?
* Mu is the set of natural images. Implications?
* Are we assuming v + ∆v<sub>i</sub> is always increasing?
    * No. We are searching for a ∆v<sub>i</sub> in a greedy way. How is previous perturbation preserved?
* Confidence of classifier while fooling?
* Correlation to pretraining on ImageNet data?
### Explore
* Control misclassification label?
* Existence of dominant labels
* Number of images or type of images required to create universal perturbations
