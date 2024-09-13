# Color Wizard ✨
#### [project ideia, still in initial stages]

"I see your true colors shining.." This project aims to give literal sense to this song by color B/W images. Feels like magic, right? Let's relive memories, One Color at a Time!

![img_color_2](https://github.com/user-attachments/assets/50463b1e-6dbb-480c-85df-f9200e57a364)

 ### Index:
- [Literature](#Literature)
  - [SOTA_Overview](#SOTA_Overview)
  - [Challenges](#Challenges)
  - [Metrics](#Metrics)
 <!--
- [Setup](#Setup)
- [Methods](#Methods)
  - [Dataset](#Dataset)
  - [Approaches](#Approaches)
  - [Results](#Results)
- [Repository_files](#Repository_files)
  -->
- [Acknowledgements](#Acknowledgements)
  
## Literature
Image Colorization is the process of adding color to B/W images. This problem is framed as being a "inverse problem" since the goal is to recover higher dimensional data (a color image) from its low-dimensional representation. Existing wotk on pixel-to-pixel problem (as in this case, since we want to predict color/label for each pixel) originally were done statistical models but currently deep neural network have been SOTA. 

  
### Challenges
- ### 1. Loss of Information
  The richness of the information capture by color imges is much whider than the grayscale images. Grayscale images, only retain the luminance information, essentially representing how bright or dark each pixel is. On the other hand, RGB images encode the world through 3 channels through a complete spectrum of light.
  
  This loss of information is the ultimate obstacle for colorization algorithms. This must be sophisticated enough so they can infer/reconstruct based on the remaining grayscale clues and possibly any additional contextual information available.

  
- ### 2. Ambiguity
  Black-and-white images represent only luminance, or brightness, which introduces ambiguity since the same shade of gray can correspond to a wide range of colors in the real world. Colorization algorithms face the challenging task of assigning colors to grayscale pixels, often relying on additional contextual information or making educated guesses to produce a realistic result.
<p align="center">
  <img src="https://github.com/user-attachments/assets/0d876afb-4f43-414b-b07a-5ca02170edfc" alt="Inherent Ambiguity" style="width:90%";>
  <br>
  <em>Inherent Ambiguity: Input Grayscale (left), Ground Truth (middle), Colorized Result (right)</em>
</p>


  
- ### 3. Lack of Semantic Understanding
  Algorithms often struggle to identiy dentify objects/materials, and their spatial relationships while coloring. Colorization algorithms may assign colors based solely on local image features, potentially leading to unrealistic color choices. For instance, it may struggle to distinguish between a brown bear and a polar bear.
<p align="center">
  <img src="https://github.com/user-attachments/assets/471d31ff-307d-431b-9f74-a53b187d3e90" alt="Semantic Understanding" style="width:90%";>
  <br>
  <em>Lack of Semantic Understanding: Input Grayscale (left), Colorized Result (middle), Ground Truth (right)</em>
</p>
  
- ### 4. Perceptual Color Constancy
  Humans have the ability of compensating for variations in lighting conditions, allowing us to perceive object colors as relatively consistent despite changes in illumination. This is know as 𝐂𝐨𝐥𝐨𝐫 𝐂𝐨𝐧𝐬𝐭𝐚𝐧𝐜𝐲 and is hard to handling/mimic in the image colorization task. Since the models need to also ensure color consistent with the perceived lighting in the scene.
<p align="center">
  <img src="https://github.com/user-attachments/assets/11350c15-4323-4711-9a54-8e8a2b242459" alt="Color Constancy" style="width:90%";>
  <br>
  <em>Perceptual Color Constancy: Input Grayscale (left), Colorized Result (middle), Ground Truth (right)</em>
</p>


- ### 5. Color Bleeding
  Color bleeding, or color inconsistency, happens in image colorization when predicted colors blend between or contaminate neighboring regions. This occurs because algorithms often rely on local image features and might struggle to distinguish between different objects with similar grayscale values, leading to unrealistic color transitions and a lack of sharp boundaries.
<p align="center">
  <img src="https://github.com/user-attachments/assets/e345d3c3-dd14-4a1d-a608-c79d27c8b209" alt="Color Inconsistency" style="width:90%";>
  <br>
  <em>Color Inconsistency: Ground Truth (left), Result w/ Bleeding (middle), Result w/o Bleeding (right)</em>
</p>


  
- ### 6. Computational Resources
   Image colorization is a computationally intensive task due to large image size, algorithm complexity, and optimization techniques.
   Larger images require more processing power, while complex deep learning models needed, involve numerous calculations and parameters. Additionally, iterative optimization processes further increase computation demands. 


### SOTA_Overview

Traditional image colorization methods include manual coloring, rule-based techniques, and color propagation approaches. However, with the advancements in deep learning, these have been largely replaced by more sophisticated methods. 

The main DL approaches are CNN-based and GAN-based colorization. Recent innovations have introduced transformer-based and diffusion-based networks, further enhancing colorization. Additionally, incorporating techniques such as object detection and semantic information has contributed significantly to advancements in the field.

- ### 1. CNNs - Colorization as a Classification task
  The paper "Colorful Image Colorization" (2016) suggest to treat this problem similary to classification and use class-rebalancing at training time to increase the diversity of colors in the result.
  <p align="center">
  <img src="https://github.com/user-attachments/assets/0f0a6c46-5883-41e2-abd6-56c9459e8831" alt="Colorful Image Colorization" style="width:90%";>
  <br>
  <em>Colorful Image Colorization (2016)</em>
  </p>
  
  In this paper authors propose an end-to-end CNN that automatically predicts vibrant/realistic colors for grayscale images.

  **A) Color Space:**
  Their approach focuses on the CIE Lab color space, where they predict the a and b channels (chromaticity) given the L channel (lightness) of an image.

  According to this convention, color can be represented using three values:
    - **L*** (luminance), which indicates perceived brightness.
    - **a*** (red — green)
    - **b*** (blue — yellow) 
    **a*** and **b*** representing the human visual colors.
    
    The **a*** and **b*** values reflect how humans perceive colors. The Lab color model is designed to be perceptually uniform, meaning that numeric changes in these values correspond to similar changes in how the colors are perceived. This feature makes it particularly useful for detecting subtle differences between colors, as it closely aligns with human vision.

  The following figure illustrates an example of the decomposition of an RGB image into the CIELAB color space:
  <p align="center">
   <img src="https://github.com/user-attachments/assets/71c84ba1-b37d-411f-b821-0e009e659d79" alt="chameleon" style="width:90%";>
   <br>
   <em>Chameleon with its decomposition into L*, a* and b* channels</em>
  </p>
  
  **B) Classification:** Instead of treating color prediction as a regression task, where the network outputs continuous color values, the color space was quantized (313 bins) and model predict the most likely bin, similar to a clssification task.

  **C) Class Rebalancing:** Since natural images contain more neural/smooth colors, the loss function was reweighted to favor rare/saturated colors/"classes".

  **D) Loss Function:** Multinomial **cross-entropy loss** is used, which compares the predicted color distribution to the true color distribution at each pixel.
  Additionally, a technique called **Annealed-Mean** was adopted, in which instead of choosing the most likely color (which could lead to spatial inconsistency), it computes the "softmax temperature" to select a color that is a compromise between the mean and mode of the predicted color distribution, balancing spatial consistency and vividness of colors.

- ### 2. GANs - Adversarial Learning Strategy
  The paper "ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution" (2020), proposes a approach to that uses GANs and semantic information. GANs can be used to generate high-quality colorizations that are more realistic, more coherent/consistent with the original b/w image and visually appealing but they need a large amount of data.

  **A) Adversarial Learning:** Similary to the previous method, this also make use of  the CIE Lab color space. Thus, ChromaGAN integrates a generator to predict the color (a, b) and a discriminator (D) to distinguish between real and fake colorizations.
  
  **B) Semantic Class Distribution:** The Generator was two outputs: 1) color (a, b); 2) class distribution vector (y) that represents the probability distribution of different semantic categories/objects in the image. Incorporating semantic information allows to generate more context-aware colorizations, improving both perceptual realism and consistency in object coloring.
  
  **C) Network Architecture:**
  - The generator has two subnetworks:
       - G1 predicts the color (channels a, b).
       - G2 predicts the class distribution vector (y) - add semantic understanding.
  - The discriminator uses the PatchGAN architecture, focusing on local patches of the image to better model high-frequency structures, resulting in sharper colorizations.
    
    <p align="center">
     <img src="https://github.com/user-attachments/assets/5f382038-44b0-4cd8-a031-0b1796c319d3" alt="ChromaGAN" style="width:90%";>
     <br>
     <em>ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution (2020)</em>
    </p>


  **D) Loss Function:**

  The objective function is define as:
  <p align="center">
   𝑳𝒐𝒔𝒔= 𝑾𝑮𝑨𝑵 𝑳𝒐𝒔𝒔 + λ𝒈 * 𝑲𝑳_𝑫𝒊𝒗𝒆𝒓𝒈𝒆𝒏𝒄𝒆_𝑪𝒍𝒂𝒔𝒔_𝑫𝒊𝒔𝒕𝒓𝒊𝒃𝒖𝒕𝒊𝒐𝒏 + λ𝒔 * 𝑪𝒐𝒍𝒐𝒓_𝑬𝒓𝒓𝒐𝒓_𝑳𝒐𝒔𝒔
  </p>
  
    - <u>𝘞𝘎𝘈𝘕 𝘓𝘰𝘴𝘴:</u> Wasserstein GAN (WGAN) minimizes the Earth-Mover distance between real and generated images, ensure more stable and realistic colorization, with a gradient penalty ensuring Lipschitz continuity in the discriminator for improved training stability.
       
    - 𝘒𝘓 𝘋𝘪𝘷𝘦𝘳𝘨𝘦𝘯𝘤𝘦 𝘊𝘭𝘢𝘴𝘴 𝘋𝘪𝘴𝘵𝘳𝘪𝘣𝘶𝘵𝘪𝘰𝘯: KL divergence to align the predicted class distribution with the VGG-16 model output, allowing the generator to learn meaningful object-level semantics for more accurate colorization
  
    - 𝘊𝘰𝘭𝘰𝘳 𝘌𝘳𝘳𝘰𝘳 𝘓𝘰𝘴𝘴: L2 norm between the predicted and real chrominance channels.


- ### 3. Incorporating Additional Information

- ### 4. Advanced Architectures




### Metrics
Evaluating the quality of image colorization is complicated, so usually it involves both objective metrics & subjective human perception. Metrics provide/measure color accuracy, consistency, and alignment with the original grayscale image. However, these metrics alone are often insufficient, as the perceived quality of colorization also depends on how realistic is the result for humans.

- ### PSNR (Peak Signal-to-Noise Ratio)
  PSNR was originaly from signal comparation and measures the peak power of a signal compared to the power of corrupting noise. However have been used to quantificar image quality and compare images.
  
  PSNR reflets the (pixel-by-pixel) difference between the original color image and the artificial colorized image. Higher PSNR indicates a lower difference, but it doesn’t necessarily guarantee perceptual similarity. PSNR can be misleading in evaluating image colorization quality because it measures overall intensity similarity, not color realism. As a result, it might indicate high quality even if the colors are unrealistic, as long as their overall intensity matches the ground truth.
<p align="center">
  <img src="https://github.com/user-attachments/assets/79dcd809-88e4-4580-9f88-f46abed12d95" alt="PSNR fails samples" style="width:80%";>
  <br>
  <em>Sample where PSNR metric fails to identify the better colorized img. 
   <br>The middle column has a better PSNR compared to the right column.</em>
</p>

- ### SSIM (Structural Similarity Index)
  SSIM (Structural Similarity Index) provides a evaluation of image quality by assessing luminance, contrast, and structural similarity between images. It captures how well the colorized image preserves the structural details of the original, offering a more nuanced assessment compared to PSNR metrics. However, despite its advanced approach, SSIM may not fully align with human perception of colorization quality.
<p align="center">
  <img src="https://github.com/user-attachments/assets/a5d32bb5-6fd5-4c4d-a9a7-99a5c4a3ad5d" alt="SSIM" style="width:80%";>
  <br>
  <em>SSIM metrics for different images structures</em>
</p>


- ### LPIPS (Learned Perceptual Image Patch Similarity)
  Uses pre-trained deep network to assess perceptual similarity between image patches. Evaluate local features in the original and colorized images, aiming to reflect human visual perception more accurately by focusing on the nuanced differences in image details.


- ### FID (Fréchet Inception Distance)
  FID evaluates the similarity between image distributions by comparing feature representations from a pre-trained network. It measures how closely the feature distributions of original and colorized images match, assuming that effective colorization will produce similar feature distributions.

  <!-- ![image](https://github.com/user-attachments/assets/4fb17395-d97c-48ab-ac68-60b3b09cfce9) -->

 <!--
## Setup
## Methods
### Dataset
### Approaches
### Results

## Repository_files
-->
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
- "Awesome Image Colorization," [MarkMoHR/Awesome-Image-Colorization](https://github.com/MarkMoHR/Awesome-Image-Colorization)
- "ColorIt," [Ye11ow-Flash/ColorIt](https://github.com/Ye11ow-Flash/ColorIt)
- "ColNet," [kainoj/colnet](https://github.com/kainoj/colnet)
- Goree, Sam. "Colorization Companion Blog," 2021. [Link](https://samgoree.github.io/2021/04/21/colorization_companion.html)
- "Overview of Image Similarity Metrics," [Medium Article by Data Monsters](https://medium.com/@datamonsters/a-quick-overview-of-methods-to-measure-the-similarity-between-images-f907166694ee)
- "Grayscale Image Colorization Methods Overview and Evaluation," [ResearchGate Publication by Various Authors](https://www.researchgate.net/publication/353854254_Grayscale_Image_Colorization_Methods_Overview_and_Evaluation)
- https://mina86.com/2021/srgb-lab-lchab-conversions/
- https://huggingface.co/tasks/image-to-image
- https://huggingface.co/models?pipeline_tag=image-to-image&sort=trending&search=color
<!--
- https://github.com/MarkMoHR/Awesome-Image-Colorization
- https://github.com/Ye11ow-Flash/ColorIt
- https://github.com/kainoj/colnet
- https://samgoree.github.io/2021/04/21/colorization_companion.html
- https://medium.com/@datamonsters/a-quick-overview-of-methods-to-measure-the-similarity-between-images-f907166694ee
- https://www.researchgate.net/publication/353854254_Grayscale_Image_Colorization_Methods_Overview_and_Evaluation
-->
