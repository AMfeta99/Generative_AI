# Color Wizard ✨
#### [project ideia, still in initial stages]

"I see your true colors shining.." This project aims to give literal sense to this song by color B/W images. Feels like magic, right? Let's relive memories, One Color at a Time!

![img_color_2](https://github.com/user-attachments/assets/50463b1e-6dbb-480c-85df-f9200e57a364)

 ### Index:
- [Literature](#Literature)
  - [Challenges](#Challenges)
  - [SOTA_Overview](#SOTA_Overview)
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
  
- ### 2. AutoEncoder
  <!-- https://github.com/alexandrasalem/image_colorization/tree/main -->
  VAE-based colorization models approach the task by encoding grayscale images into a compressed latent space that captures both semantic and color information. The latent representation is then decoded to predict the chrominance channels (a, b). This method allows for multiple plausible colorizations since different points in the latent space can correspond to various valid color combinations. Colorization in the context of VAE is treated as a supervised learning problem, where the VAE learns through quadratic regression on a dataset of color images.

  VAE-based models often face the challenge of generating **blurry images** due to the likelihood-based loss functions used, such as L2 or reconstruction loss. Since these losses do not effectively capture high-frequency details like textures and sharp color boundaries. Although VAEs generate diverse and plausible colorizations, colorized images may lack the sharpness and clarity that adversarial training in more sophisticated approaches like GANs can achieve.

- ### 3. GANs - Adversarial Learning Strategy
  GANs can be used to generate high-quality colorizations that are more realistic, more coherent/consistent with the original b/w image and visually appealing but they need a large amount of data. Some notable GAN-based colorization approaches:
    <p align="center">
   <img src="https://github.com/user-attachments/assets/692220db-bae1-44b2-a7d7-63a8fc4123e8" alt="cGANs" style="width:50%";>
   <br>
   <em>Conditiona GANs for image Colorization</em>
   </p>

    - **Isola et al. (2017):** They introduced conditional GANs (cGANs), which map grayscale images to colored ones using paired data. They trained the generator to predict the chrominance values (a, b) conditioned on the grayscale image (L) and combined the adversarial loss (GAN loss) with an L1 loss for better results. This approach outputs sharp, realistic colorizations by leveraging both the low-level reconstruction error (L1 loss) and high-level perceptual quality (GAN loss).
      
    - **Cao et al. (2017):**  This method used a conditional GAN architecture but also incorporated input noise into multiple layers of the network. This noise allowed the model to generate diverse colorizations for the same grayscale image by sampling different noise values. Their fully convolutional non-stride network enabled fine control over the colorization process.
      
    - **Nazeri and Ng (2018):** In an extension of Isola’s method, this approach aimed to generalize the model for high-resolution images. The GAN loss was optimized for better stability and speed during training. High-resolution colorization required more sophisticated training techniques to prevent issues like mode collapse and vanishing gradients.
 
    - **He et al. (2018):** Propose a exemplar-based GANs that use reference images to guide local and plausible color prediction, allowing for more flexible and accurate colorization outcomes.
      

  Recently, 2020, Vitoria et al. publish **"ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution"**, proposing a approach to that uses GANs and **semantic information**.
  <p align="center">
     <img src="https://github.com/user-attachments/assets/5f382038-44b0-4cd8-a031-0b1796c319d3" alt="ChromaGAN" style="width:90%";>
     <br>
     <em>ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution (2020)</em>
  </p>


  **A) Adversarial Learning:** Similary to the previous method, this also make use of  the CIE Lab color space. Thus, ChromaGAN integrates a generator to predict the color (a, b) and a discriminator (D) to distinguish between real and fake colorizations.
  
  **B) Semantic Class Distribution:** The Generator was two outputs: 1) color (a, b); 2) class distribution vector (y) that represents the probability distribution of different semantic categories/objects in the image. Incorporating semantic information allows to generate more context-aware colorizations, improving both perceptual realism and consistency in object coloring.
  
  **C) Network Architecture:**
  - The generator has two subnetworks:
       - G1 predicts the color (channels a, b).
       - G2 predicts the class distribution vector (y) - add semantic understanding.
  - The discriminator uses the PatchGAN architecture, focusing on local patches of the image to better model high-frequency structures, resulting in sharper colorizations.
    

  **D) Loss Function:**

  The objective function is define as:
  <p align="center">
   𝑳𝒐𝒔𝒔= 𝑾𝑮𝑨𝑵 𝑳𝒐𝒔𝒔 + λ𝒈 * 𝑲𝑳_𝑫𝒊𝒗𝒆𝒓𝒈𝒆𝒏𝒄𝒆_𝑪𝒍𝒂𝒔𝒔_𝑫𝒊𝒔𝒕𝒓𝒊𝒃𝒖𝒕𝒊𝒐𝒏 + λ𝒔 * 𝑪𝒐𝒍𝒐𝒓_𝑬𝒓𝒓𝒐𝒓_𝑳𝒐𝒔𝒔
  </p>
  
    - W̳G̳A̳N̳ ̳L̳o̳s̳s̳: Wasserstein GAN (WGAN) minimizes the Earth-Mover distance between real and generated images, ensure more stable and realistic colorization, with a gradient penalty ensuring Lipschitz continuity in the discriminator for improved training stability.
       
    - ̳K̳L̳ ̳D̳i̳v̳e̳r̳g̳e̳n̳c̳e̳ ̳C̳l̳a̳s̳s̳ ̳D̳i̳s̳t̳r̳i̳b̳u̳t̳i̳o̳n̳: KL divergence to align the predicted class distribution with the VGG-16 model output, allowing the generator to learn meaningful object-level semantics for more accurate colorization
  
    - C̳o̳l̳o̳r̳ ̳E̳r̳r̳o̳r̳ ̳L̳o̳s̳s̳:  L2 norm between the predicted and real chrominance channels.
      

Since 2020, several advanced versions and improvements using GANs have been proposed, such as SCGANs, CycleGAN, and even multi-GAN approaches. However, a notable concern in this area is incorporating additional semantic information into the algorithms.
      
- ### 4. Incorporating Additional Information
  **A) Object Detection Information**
  
  One major challenge in coloration is handling complex scenes with various objects, often algorithms struggle assigning colors into neighboring regions and create unrealistic artifacts.
  
  Recent approaches tackle this by using object detection models to identify and separate objects within a grayscale image. This object information is then used to guide the colorization process, ensuring colors are applied more accurately. This method creates sharper boundaries between objects and produces a more natural, realistic result.
  <p align="center">
  <img src="https://github.com/user-attachments/assets/ee315d34-c4e8-4853-a825-22ddfa401bc3" alt="Instance-aware Image Colorization" style="width:80%";>
  <br>
  <em>Instance-aware Image Colorization (2020)</em>
  </p>
   The paper "Instance-Aware Image Colorization" overcome this by detecting/segment objects in B/W img and colorize each of them individually. To do this separate networks are used to handle object-level and scene-level colorization as described:
 
    - **Object Detection:** The model first identifies and segments objects within the image, which provides a clear separation between different objects (like cars, trees, benches).
    - **Feature Extraction:** Separate networks then extract features both at the object level (focusing on the properties of the specific object) and the image level (considering the overall scene).
    - **Feature Fusion:** These two sets of features—object-specific and scene-wide—are intelligently combined. This fusion allows the model to make better color predictions based on both the object's likely color and the scene context.
 
  **B) Semantic Information**
  
  Understanding the overall context of an image is key to realistic colorization. Researchers now use scene segmentation to extract detailed labels, **identifying** not just objects but also **materials** (e.g., wood, grass) and **scene elements** (e.g., sky, ground).

  This richer semantic understanding helps colorization models make more accurate and context-aware color decisions, leading to more realistic and consistent results. 
  <p align="center">
  <img src="https://github.com/user-attachments/assets/1432f7e4-3bf7-449f-acdb-073c5e460aa1" alt="Pixelated Semantic Colorization" style="width:80%";>
  <br>
  <em>Pixelated Semantic Colorization (2020)</em></p>
  
  The paper "Pixelated Semantic Colorization" explores how understanding the semantic content of an image can improve colorization. It proposes a two-branch network that combines semantic understanding with colorization:
  
    - **Semantic Understanding Branch:** This part of the model identifies objects and scene elements using techniques like semantic segmentation, understanding what’s in the image.
    - **Pixel-Level Semantic Embedding:** The semantic information is transformed into a detailed pixel-level embedding, capturing the meaning of each pixel in the image.
    - **Colorization Guided by Semantics:** This semantic embedding is then used in the colorization process, helping the model predict appropriate colors based on the object or scene (e.g., making the sky blue).


- ### 5. Advanced Architectures
  **A) Diffusion-based Network**
  
    Diffusion models, known for generating realistic images by refining noise, are now being adapted for colorization.
  
    In this approach, the process starts with a grayscale image, and the model gradually adds color through an iterative refinement process. This method has great potential to produce high-quality colorizations, capturing subtle variations and details, leading to more realistic results.
  <p align="center">
   <img src="https://github.com/user-attachments/assets/94b72b61-3335-4662-b60d-82460e8c47f0" alt="diffusion-based" style="width:80%";>
   <br>
   <em>Improved Diffusion-based Image Colorization via Piggybacked Models (2023)</em></p>

   The paper "Improved Diffusion-based Image Colorization via Piggybacked Models" introduces a novel colorization method using **diffusion models**. It leverages **pre-trained text-to-image diffusion models**, which are typically trained to generate images from text. By tapping into the model's knowledge of color and its **link to semantic concepts**, the process achieves more realistic colorization.

  **Key components:**
  
     - **Diffusion Guider for Color Priors:** This extracts color information from the pre-trained model, aligning it with the grayscale image's content to create a color foundation.
     - **Lightness-Aware VQ-VAE:** This architecture, aware of lightness details in the image, combines the grayscale input and color prior to generate the final colorized image, ensuring pixel-perfect alignment and avoiding artifacts.
  
    This approach benefits from the diffusion model's understanding of color semantics and allows for conditional colorization, where user can guide color choices.
  

  **B) Transformer-based Network**
  
   CNNs struggle with capturing long-range dependencies in images. To overcome this, researchers are turning to transformer architectures, which can analyze the entire B/W image at once. This allows transformers to consider the global context when predicting colors for each pixel, leading to better color harmony and consistency across the image. Transformer-based models offer an advantage over CNNs by capturing these broader relationships within an image.

  The papers **ColTran: "Colorization Transformer"** (2021) and **"CT²: Colorization Transformer via Color Tokens"** (2022) both introduce innovative transformer-based approaches to image colorization, moving beyond traditional CNN methods.

    - **ColTran:** ColTran uses a progressive approach, starting with a coarse, low-resolution colorization and gradually refining the colors and resolution step-by-step. It leverages axial transformers, which capture spatial relationships across both horizontal and vertical directions. This method focuses on achieving an overall color scheme first, then adding finer details for natural-looking colorizations.
      <p align="center">
       <img src="https://github.com/user-attachments/assets/30fa3ff9-713e-4bc8-9a9c-adf4f3cafe6e" alt="ColTran" style="width:80%";>
       <br>
       <em>Colorization Transformer (ColTran) </em></p>

      
    - **CT²:** CT² addresses a common issue in transformer-based colorization—undersaturation—by introducing color tokens. These tokens represent discrete colors in a predefined color space. Instead of directly predicting RGB values, CT² treats colorization as a classification problem, assigning probability scores to each token for each pixel. This approach ensures more saturated, vibrant color results while limiting the palette to realistic, diverse color choices.
      <p align="center">
       <img src="https://github.com/user-attachments/assets/0480effb-72a6-4ade-bb5f-a4ee7443db2a" alt="CT2" style="width:80%";>
       <br>
       <em>Colorization Transformer via Color Tokens (CT²) </em></p>

  Both methods highlight the power of transformers in achieving more globally consistent and vibrant colorizations compared to CNNs.


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

1. Adrian Rosebrock, "Black and White Image Colorization with OpenCV and Deep Learning," PyImageSearch, February 25, 2019. [Link](https://pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/)
2. Weichen Pai, "Image Colorization: Bringing Black and White to Life," Medium, August 8, 2020. [Link](https://medium.com/@weichenpai/image-colorization-bringing-black-and-white-to-life-b14d3e0db763)
3. [HuggingFace Colorization Datasets](https://huggingface.co/datasets?search=colorization): A collection of datasets available for image colorization tasks.
4. Zhang, Richard, et al. "Colorization Using CNN" . [Link](https://richzhang.github.io/colorization/) (CNN approach).
5. Vitoria, Patrícia, et al. "ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution," WACV 2020. [Link](https://openaccess.thecvf.com/content_WACV_2020/papers/Vitoria_ChromaGAN_Adversarial_Picture_Colorization_with_Semantic_Class_Distribution_WACV_2020_paper.pdf) (GANs Approach)
6. Su, Jian, et al. "Instance-Aware Image Colorization," CVPR 2020. [Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Instance-Aware_Image_Colorization_CVPR_2020_paper.pdf) (Tackles lack of semantic understanding and handles complex scenes with object detection).
7. Iizuka, Satoshi, et al. "Colorizing Images While Understanding Their Semantic Content," International Journal of Computer Vision (IJCV), 2019. [Link](https://link.springer.com/article/10.1007/s11263-019-01271-4)
8. Kumar, Ashish, et al. "Colorization Using Transformer," 2021. [Link](https://arxiv.org/pdf/2102.04432.pdf) (Transformer-based approach).
9. "Colorization with Deep Learning," Springer 2022. [Link](https://link.springer.com/chapter/10.1007/978-3-031-20071-7_1)
10. Li, Xiang, et al. "Improved Diffusion-based Image Colorization via Piggybacked Models," 2023. [Link](https://arxiv.org/pdf/2304.11105.pdf) (Diffusion-based approach)
11. "Perceptual Similarity," A metric to compare images based on human perception. [Link](https://wiki.spencerwoo.com/perceptual-similarity.html)
12. "Awesome Image Colorization," [MarkMoHR/Awesome-Image-Colorization](https://github.com/MarkMoHR/Awesome-Image-Colorization)
13. "ColorIt," [Ye11ow-Flash/ColorIt](https://github.com/Ye11ow-Flash/ColorIt)
14. "ColNet," [kainoj/colnet](https://github.com/kainoj/colnet)
15. Goree, Sam. "Colorization Companion Blog," 2021. [Link](https://samgoree.github.io/2021/04/21/colorization_companion.html)
16. "Overview of Image Similarity Metrics," [Medium Article by Data Monsters](https://medium.com/@datamonsters/a-quick-overview-of-methods-to-measure-the-similarity-between-images-f907166694ee)
17. "Grayscale Image Colorization Methods Overview and Evaluation," [ResearchGate Publication by Various Authors](https://www.researchgate.net/publication/353854254_Grayscale_Image_Colorization_Methods_Overview_and_Evaluation)
18. mina86. "sRGB, Lab and LCh(ab) Conversions," 2021. [Link](https://mina86.com/2021/srgb-lab-lchab-conversions/)
19. Hugging Face. "Image-to-Image Tasks," Hugging Face, 2021. [Link](https://huggingface.co/tasks/image-to-image)
20. Hugging Face. "Trending Image-to-Image Models," Hugging Face, 2021. [Link](https://huggingface.co/models?pipeline_tag=image-to-image&sort=trending&search=color)
21. SpringerLink. "Colorization and Image Translation," *Encyclopedia of Computer Graphics and Games*, 2022. [Link](https://link.springer.com/referenceworkentry/10.1007/978-3-030-98661-2_55)
22. SpringerLink. "Figure from Colorization Research," *Artificial Intelligence-Based Cancer Prediction for Socioeconomic Development*, 2021. [Link](https://link.springer.com/chapter/10.1007/978-981-16-0708-0_2/figures/3)
23. Zhang, Richard et al. "Learning Representations for Image Colorization," *International Journal of Computer Vision*, 2019. [Link](https://link.springer.com/article/10.1007/s11263-019-01271-4)
24. Mordvintsev, Alexander et al. "Interactive Deep Colorization of Grayscale Images," *arXiv preprint*, 2021. [Link](https://arxiv.org/pdf/2102.04432.pdf)
25. Su, Haotian et al. "Instance-Aware Image Colorization," *CVPR Conference*, 2020. [Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Instance-Aware_Image_Colorization_CVPR_2020_paper.pdf)
26. Ender, Rebecca et al. "Image Colorization with Multimodal Models," *arXiv preprint*, 2023. [Link](https://arxiv.org/pdf/2304.11105.pdf)


