# Magic Recipe
[project ideia, still in initial stages]

<!-- Walt Disney used to say, 'If you can dream it, you can do it!'
I suggest a slight change to this famous quote: 'If you can describe it, we can do it!😜' 

This repository aims to experiment with SOTA methods for tex-to-img task. This journey is also a great way to develop skills in popular platforms and tools such as Hugging Face, OpenCV, Diffusers, PyTorch, Prompt Engineering, NLP.
<p align="center">
  <img src="https://github.com/user-attachments/assets/abf2b2c1-48d3-4c6d-b3c1-643dc3e45116" alt="Dalli" style="width:70%";>
  <br>
  <em></em>
</p>
-->
<!-- ![image](https://github.com/user-attachments/assets/abf2b2c1-48d3-4c6d-b3c1-643dc3e45116) -->

<!-- ver esta ideia ! https://www.youtube.com/watch?v=FMRi6pNAoag -->
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
**Text-to-image** models are a type of AI that can take a text description and create an image that match the description. These models work by using a language model to understand the text and then a generative model to create an image based on that understanding. Examples models for the task include **DALL-E 2**, **GLIDE** and **Imagen**. Imagen, for instance, is a model developed by Google in 2022 that uses text encodings to guide image generation.

These models are trained on huge amounts of text and images from the internet, which helps them learn how to connect words with visual features. They’ve become so advanced that they can create very realistic images.

### SOTA_Overview

- ### 1. AutoEncoder
- ### 2. Diffusion Models
  Diffusion models are a type of generative AI used in text-to-image tasks. The core idea behind these models is to gradually remove noise from an image to reveal its structure. Here's how they work:

   **A) Text Encoding:**
  The input text is first transformed into a "latent representation" (a numerical form that captures the meaning of the text) using a text encoder.

  **B) Diffusion Process:**
  Starting with a noisy image, the model slowly removes the noise over multiple steps. This process is guided by the text's latent representation, helping the model generate an image that matches the description.

  **C) Diffusion Operator:**
  At each step, a "diffusion operator" cleans up some noise from the image, getting closer to the final result. This process is repeated for a fixed number of steps until the final image is formed.

<!-- Mehtods to implement
https://abdulkaderhelwan.medium.com/text-to-image-generation-model-with-cnn-ca904427d1e7
-->
<!--
repo img: https://aiimagegenerator.in/
https://huggingface.co/tasks/text-to-image
https://abdulkaderhelwan.medium.com/text-to-image-generation-model-with-cnn-ca904427d1e7
https://www.kaggle.com/code/stpeteishii/text-to-image-generation-by-stable-diffusion

# review
https://arxiv.org/pdf/2303.07909
https://www.bentoml.com/blog/a-guide-to-open-source-image-generation-models  (imp)
https://dominguezdaniel.medium.com/exploring-image-generative-ai-models-9359705b15d3
https://medium.com/@aisagescribe/generative-ai-for-image-generation-sota-common-methods-bea0a70c9b81
https://www.codementor.io/@kalpesh08/how-does-ai-turn-text-into-images-2amzolymx5
https://arxiv.org/pdf/2309.00810
https://arxiv.org/pdf/2303.07909
-->

