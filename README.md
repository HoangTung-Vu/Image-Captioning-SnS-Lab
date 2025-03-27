# Image Captioning Project

This project focuses on generating descriptive captions for images by leveraging deep learning techniques, combining computer vision and natural language processing methodologies.

## Project Structure

The repository is organized into the following directories:

- `model/`: Contains the architecture and training scripts for the image captioning model.
- `scripts/`: Includes utility scripts for data preprocessing and evaluation.
- `utils/`: Provides helper functions to support various aspects of the project.

## Approach

The image captioning model employs an encoder-decoder architecture:

1. **Encoder**: A convolutional neural network (CNN) extracts feature representations from input images.
2. **Decoder**: A recurrent neural network (RNN), such as Long Short-Term Memory (LSTM), generates textual descriptions based on the encoded image features.

This approach aligns with methodologies discussed in related projects, such as the implementation of a Neural Image Caption (NIC) network for generating image captions

## Datasets

Commonly used datasets for image captioning tasks include:

- **MS COCO**: A large-scale dataset for image recognition, segmentation, and captioning.
- **Flickr30K**: A dataset comprising 30,000 images with corresponding captions collected from Flickr.
- **Flickr8K**: Consists of 8,000 images, each annotated with five different captions.

These datasets are widely utilized in the field of image captioning.

## Evaluation Metrics

To assess the performance of the image captioning model, the following metrics are commonly employed:

- **BLEU (Bilingual Evaluation Understudy)**: Evaluates the precision of n-grams in the generated captions compared to reference captions.
- **CIDEr (Consensus-based Image Description Evaluation)**: Measures the similarity of generated captions to human-written captions based on term frequency-inverse document frequency (TF-IDF) weighting.
- **METEOR (Metric for Evaluation of Translation with Explicit ORdering)**: Considers precision, recall, and alignment of words between generated and reference captions.

These metrics are standard in evaluating image captioning models. citeturn0search4

## Getting Started

To set up and run the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HoangTung-Vu/Image-Captioning-SnS-Lab.git
   ```
2. **Install dependencies**: Ensure that all required packages are installed as specified in the `requirements.txt` file.
3. **Prepare the dataset**: Download and preprocess the chosen dataset (e.g., MS COCO, Flickr30K) using the scripts provided in the `scripts/` directory.
4. **Train the model**: Execute the training script located in the `model/` directory to train the image captioning model.
5. **Evaluate the model**: Use the evaluation scripts to assess the performance of the trained model using the aforementioned metrics.

## Acknowledgments

This project is inspired by various works in the field of image captioning, including implementations and methodologies from related projects. citeturn0search1

--- 
