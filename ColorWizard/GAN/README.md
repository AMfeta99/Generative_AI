# GANs-Based Approach 

**Overview**
Generative Adversarial Networks consist of two neural networks: a Generator (G) and a Discriminator (D), which compete against each other in a zero-sum game.



## Repository_files
- models: store  model architectures.
- datasets: Handle data loading and preprocessing.
- utils: Include helper functions (e.g., evaluation metrics, logging).
- train.py:  Train the models and save checkpoints.
- config.py:  Manage hyperparameters and paths.


## Models
- Generator to generate fake data (images, text, etc.).
- Discriminator to distinguish between real and fake data.

## Training Loop
- Train D to maximize the probability of correctly classifying real and fake samples.
- Train G to minimize D's ability to classify fake samples.
- Gradient updates using optimizers like Adam.
- Monitor convergence using metrics like GAN loss and visual inspection.

## Hyperparameters
- Batch size, learning rates for G and D, latent vector size, etc. (config.py).

<!--
## Setup
## Methods
Hugging face (https://huggingface.co/models?pipeline_tag=image-to-image&sort=trending&search=colo)
https://huggingface.co/c2p-cmd/bw_colorizer_coreml
https://huggingface.co/rsortino/ColorizeNet
https://huggingface.co/Hammad712/GAN-Colorization-Model
https://huggingface.co/TencentARC/t2iadapter_color_sd14v1

github 
https://github.com/MarkMoHR/Awesome-Image-Colorization (principal... all papers and code)
https://github.com/richzhang/colorization-pytorch (cycle-GAN)
https://github.com/ericsujw/InstColorization/tree/master (same as before but with object detection)
https://github.com/msracver/Deep-Exemplar-based-Colorization
https://github.com/zeruniverse/neural-colorization  (GAN)



## Acknowledgements

1. Zhang, Richard, et al. "Colorization Using CNN." [Link](https://richzhang.github.io/colorization/) (CNN approach). (No specific date)
2. Iizuka, Satoshi, et al. "Colorizing Images While Understanding Their Semantic Content," International Journal of Computer Vision (IJCV), 2019. [Link](https://link.springer.com/article/10.1007/s11263-019-01271-4)
3. Adrian Rosebrock, "Black and White Image Colorization with OpenCV and Deep Learning," PyImageSearch, February 25, 2019. [Link](https://pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/)
4. Vitoria, Patrícia, et al. "ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution," WACV 2020. [Link](https://openaccess.thecvf.com/content_WACV_2020/papers/Vitoria_ChromaGAN_Adversarial_Picture_Colorization_with_Semantic_Class_Distribution_WACV_2020_paper.pdf) (GANs Approach)
5. Su, Jian, et al. "Instance-Aware Image Colorization," CVPR 2020. [Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Instance-Aware_Image_Colorization_CVPR_2020_paper.pdf) (Tackles lack of semantic understanding and handles complex scenes with object detection).
6. Weichen Pai, "Image Colorization: Bringing Black and White to Life," Medium, August 8, 2020. [Link](https://medium.com/@weichenpai/image-colorization-bringing-black-and-white-to-life-b14d3e0db763)
7. Kumar, Ashish, et al. "Colorization Using Transformer," 2021. [Link](https://arxiv.org/pdf/2102.04432.pdf) (Transformer-based approach).
8. Li, Xiang, et al. "Improved Diffusion-based Image Colorization via Piggybacked Models," 2023. [Link](https://arxiv.org/pdf/2304.11105.pdf) (Diffusion-based approach)
9. Ender, Rebecca et al. "Image Colorization with Multimodal Models," *arXiv preprint*, 2023. [Link](https://arxiv.org/pdf/2304.11105.pdf)
10. "Colorization with Deep Learning," Springer 2022. [Link](https://link.springer.com/chapter/10.1007/978-3-031-20071-7_1)
11. "AI in Image Colorization Techniques," SpringerLink, *Artificial Intelligence for Medical Image Analysis*, 2023. [Link](https://link.springer.com/chapter/10.1007/978-3-031-20071-7_1)
12. Mordvintsev, Alexander et al. "Interactive Deep Colorization of Grayscale Images," *arXiv preprint*, 2021. [Link](https://arxiv.org/pdf/2102.04432.pdf)
13. Su, Haotian et al. "Instance-Aware Image Colorization," *CVPR Conference*, 2020. [Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Instance-Aware_Image_Colorization_CVPR_2020_paper.pdf)
14. SpringerLink. "Figure from Colorization Research," *Artificial Intelligence-Based Cancer Prediction for Socioeconomic Development*, 2021. [Link](https://link.springer.com/chapter/10.1007/978-981-16-0708-0_2/figures/3)
15. Zhang, Richard et al. "Learning Representations for Image Colorization," *International Journal of Computer Vision*, 2019. [Link](https://link.springer.com/article/10.1007/s11263-019-01271-4)
16. "Grayscale Image Colorization Methods Overview and Evaluation," [ResearchGate Publication by Various Authors](https://www.researchgate.net/publication/353854254_Grayscale_Image_Colorization_Methods_Overview_and_Evaluation)
17. "Perceptual Similarity," A metric to compare images based on human perception. [Link](https://wiki.spencerwoo.com/perceptual-similarity.html)
18. mina86. "sRGB, Lab and LCh(ab) Conversions," 2021. [Link](https://mina86.com/2021/srgb-lab-lchab-conversions/)
19. Hugging Face. "Image-to-Image Tasks," Hugging Face, 2021. [Link](https://huggingface.co/tasks/image-to-image)
20. Hugging Face. "Trending Image-to-Image Models," Hugging Face, 2021. [Link](https://huggingface.co/models?pipeline_tag=image-to-image&sort=trending&search=color)
21. SpringerLink. "Colorization and Image Translation," *Encyclopedia of Computer Graphics and Games*, 2022. [Link](https://link.springer.com/referenceworkentry/10.1007/978-3-030-98661-2_55)
22. "Awesome Image Colorization," [MarkMoHR/Awesome-Image-Colorization](https://github.com/MarkMoHR/Awesome-Image-Colorization)
23. "ColorIt," [Ye11ow-Flash/ColorIt](https://github.com/Ye11ow-Flash/ColorIt)
24. "ColNet," [kainoj/colnet](https://github.com/kainoj/colnet)
25. Goree, Sam. "Colorization Companion Blog," 2021. [Link](https://samgoree.github.io/2021/04/21/colorization_companion.html)
26. "Overview of Image Similarity Metrics," [Medium Article by Data Monsters](https://medium.com/@datamonsters/a-quick-overview-of-methods-to-measure-the-similarity-between-images-f907166694ee)

-->

