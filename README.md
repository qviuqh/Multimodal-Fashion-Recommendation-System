# Multimodal Fashion Recommendation System

## 📌 Project Overview
Recommender systems in the fashion domain face unique challenges due to the visually rich and subjective nature of user preferences. Traditional unimodal systems (text-only or image-only) often fail to capture visual aesthetics and model complex semantic-visual relationships. 

This project explores **multimodal recommendation models** that integrate both visual and textual features using deep learning techniques to enhance personalized fashion product discovery. Specifically, we investigate and compare feature fusion-based hybrid models with pretrained vision-language models.

This project was developed for the Computer Vision course (2025) at **National Economics University**.

## 🎯 Objectives & Research Questions
* **Design and implement** multimodal recommendation models leveraging both text and image data.
* **Compare** different multimodal learning strategies:
    * Feature fusion-based hybrid models (**ResNet50 + BERT**).
    * Pretrained vision-language models (**CLIP**).
* **Evaluate** how these strategies affect fashion recommendation performance and address challenges like cold-start items and data sparsity.

## 📊 Dataset
The project utilizes the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) from Kaggle.
* **Attributes**: High-resolution product images, rich textual descriptions, display names, and categorical labels.
* **Size**: ~44,441 fashion items.

## 🏗️ Model Architectures

### 1. Hybrid Two-Tower Model (ResNet50 + BERT)
* **Image Tower**: Pretrained `ResNet50` used as a visual feature extractor.
* **Text Tower**: Pretrained `BERT` (`bert-base-uncased`) used to encode product names and descriptions.
* **Fusion Strategy**: Late fusion. The outputs from both towers are concatenated and passed through fully connected projection layers to map them into a shared embedding space.
* **Loss Function**: Batch Hard Triplet Loss (optimizing the hardest positive and negative pairs within a batch).

### 2. Vision-Language Model (CLIP)
* **Architecture**: `openai/clip-vit-base-patch32`.
* **Approach**: Fine-tuning the CLIP model to align image and text representations in a shared semantic space specifically for the fashion domain. 
* **Advanced Techniques**: Utilized custom loss blending (Supervised Contrastive Loss + standard Contrastive Loss) and partial layer unfreezing for optimal fine-tuning.

## ⚙️ Repository Structure
* `cv-resnet-bert.ipynb`: Source code for building, training, and evaluating the ResNet50 + BERT hybrid model.
* `clip-resys-fine-tune-all.ipynb`: Source code for the standard fine-tuning process of the CLIP model.
* `clip-resys-final.ipynb`: Source code for advanced CLIP fine-tuning (includes Stratified sampling, SupCon loss, and t-SNE visualization).
* `computer_vision.pdf` / `Multimodal Fashion Recommendation System.pdf`: Research paper and presentation slides detailing the methodology, experiments, and quantitative results.

## 🚀 Installation & Usage

### Prerequisites
Make sure you have the following dependencies installed:
```bash
pip install torch torchvision transformers pandas numpy scikit-learn matplotlib seaborn tqdm
```

### Running the Code
1. The models are implemented in Jupyter Notebooks. To run the experiments:
2. Download the dataset from Kaggle and place it in the appropriate directory (e.g., /kaggle/input/fashion-product-text-images-dataset/).
3. Open any of the .ipynb notebooks.
4. Update the Config dataclass within the notebooks to point to your local dataset paths.
5. Execute the cells sequentially to train the models and extract embeddings.

## 📈 Evaluation Metrics
The models are evaluated based on their retrieval and recommendation performance using the following metrics:
* Recall@K (K = 1, 5, 10)
* MRR (Mean Reciprocal Rank)
* NDCG@10 (Normalized Discounted Cumulative Gain)

Visualizations including Training Loss curves, t-SNE Embeddings (for top categories), and Image-Text Retrieval Examples are generated at the end of the evaluation phase.
