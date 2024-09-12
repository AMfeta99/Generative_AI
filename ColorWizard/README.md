# Color Wizard ✨
[project ideia, still in initial stages]

"I see your true colors shining.." This project aims to give literal sense to this song by color B/W images. Feels like magic, right? Let's relive memories, One Color at a Time!

![img_color_2](https://github.com/user-attachments/assets/50463b1e-6dbb-480c-85df-f9200e57a364)

 ### Index:
- [Literature](#Literature)
  - [SOTA_Overview](#SOTA_Overview)
  - [Challenges](#Challenges)
  - [Metrics](#Metrics)
- [Setup](#Setup)
- [Methods](#Methods)
  - [Dataset](#Dataset)
  - [Approaches](#Approaches)
  - [Results](#Results)
- [Repository_files](#Repository_files)
- [Acknowledgements](#Acknowledgements)
  
## Literature
Image Colorization is the process of adding color to B/W images. This problem is framed as being a "inverse problem" since the goal is to recover higher dimensional data (a color image) from its low-dimensional representation. Existing wotk on pixel-to-pixel problem (as in this case, since we want to predict color/label for each pixel) originally were done statistical models but currently deep neural network have been SOTA. 

### SOTA_Overview

### Challenges
- ### 1. Loss of Information
  The richness of the information capture by color imges is much whider than the grayscale images. Grayscale images, only retain the luminance information, essentially representing how bright or dark each pixel is. On the other hand, RGB images encode the world through 3 channels through a complete spectrum of light.
  
  This loss of information is the ultimate obstacle for colorization algorithms. This must be sophisticated enough so they can infer/reconstruct based on the remaining grayscale clues and possibly any additional contextual information available.
  
- ### 2. Ambiguity
  Black-and-white images represent only luminance, or brightness, which introduces ambiguity since the same shade of gray can correspond to a wide range of colors in the real world. Colorization algorithms face the challenging task of assigning colors to grayscale pixels, often relying on additional contextual information or making educated guesses to produce a realistic result.
  
- ### 3. Lack of Semantic Understanding
  Algorithms often struggle to identiy dentify objects/materials, and their spatial relationships while coloring. Colorization algorithms may assign colors based solely on local image features, potentially leading to unrealistic color choices. For instance, it may struggle to distinguish between a brown bear and a polar bear.
  
- ### 4. Perceptual Color Constancy
  Humans have the ability of compensating for variations in lighting conditions, allowing us to perceive object colors as relatively consistent despite changes in illumination. This is know as 𝐂𝐨𝐥𝐨𝐫 𝐂𝐨𝐧𝐬𝐭𝐚𝐧𝐜𝐲 and is hard to handling/mimic in the image colorization task. Since the models need to also ensure color consistent with the perceived lighting in the scene.

- ### 5. Color Bleeding
  Color bleeding, or color inconsistency, happens in image colorization when predicted colors blend between or contaminate neighboring regions. This occurs because algorithms often rely on local image features and might struggle to distinguish between different objects with similar grayscale values, leading to unrealistic color transitions and a lack of sharp boundaries.
  
- ### 6. Computational Resources
   Image colorization is a computationally intensive task due to large image size, algorithm complexity, and optimization techniques.
   Larger images require more processing power, while complex deep learning models needed, involve numerous calculations and parameters. Additionally, iterative optimization processes further increase computation demands. 

### Metrics
PSNR (Peak Signal-to-Noise Ratio)
SSIM (Structural Similarity Index)
LPIPS (Learned Perceptual Image Patch Similarity)
FID (Fréchet Inception Distance)

## Setup
## Methods
### Dataset
### Approaches
### Results

## Repository_files

## Acknowledgements

- Adrian Rosebrock, "Black and White Image Colorization with OpenCV and Deep Learning," PyImageSearch, February 25, 2019. [Link](https://pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/)

- Weichen Pai, "Image Colorization: Bringing Black and White to Life," Medium, August 8, 2020. [Link](https://medium.com/@weichenpai/image-colorization-bringing-black-and-white-to-life-b14d3e0db763)

- [HuggingFace Colorization Datasets](https://huggingface.co/datasets?search=colorization): A collection of datasets available for image colorization tasks.
  
- Zhang, Richard, et al. "Colorization Using CNN" . [Link](https://richzhang.github.io/colorization/) (CNN approach).
  
- Vitoria, Patrícia, et al. "ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution," WACV 2020. [Link](https://openaccess.thecvf.com/content_WACV_2020/papers/Vitoria_ChromaGAN_Adversarial_Picture_Colorization_with_Semantic_Class_Distribution_WACV_2020_paper.pdf) (GANs Approach)

- Su, Jian, et al. "Instance-Aware Image Colorization," CVPR 2020. [Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Instance-Aware_Image_Colorization_CVPR_2020_paper.pdf) (Tackles lack of semantic understanding and handles complex scenes with object detection).

- Iizuka, Satoshi, et al. "Colorizing Images While Understanding Their Semantic Content," International Journal of Computer Vision (IJCV), 2019. [Link](https://link.springer.com/article/10.1007/s11263-019-01271-4)

- Kumar, Ashish, et al. "Colorization Using Transformer," 2021. [Link](https://arxiv.org/pdf/2102.04432.pdf) (Transformer-based approach).

- "Colorization with Deep Learning," Springer 2022. [Link](https://link.springer.com/chapter/10.1007/978-3-031-20071-7_1)

- Li, Xiang, et al. "Improved Diffusion-based Image Colorization via Piggybacked Models," 2023. [Link](https://arxiv.org/pdf/2304.11105.pdf) (Diffusion-based approach)
  
- "Perceptual Similarity," A metric to compare images based on human perception. [Link](https://wiki.spencerwoo.com/perceptual-similarity.html)
- https://github.com/MarkMoHR/Awesome-Image-Colorization
- https://github.com/Ye11ow-Flash/ColorIt
- https://github.com/kainoj/colnet
- https://samgoree.github.io/2021/04/21/colorization_companion.html
- https://www.researchgate.net/publication/353854254_Grayscale_Image_Colorization_Methods_Overview_and_Evaluation
