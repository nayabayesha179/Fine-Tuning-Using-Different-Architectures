# Fine-Tuning-Using-Different-Architectures
Encoder Model (BERT), Decoder Model (GPT-2), Encoder-Decoder Model (T5)

Task1: Emotion Classification with BERT

This project implements a fine-tuned BERT model for emotion classification, trained to identify four emotional categories: Anger, Joy, Neutral, and Sadness from text data
📋 Project Overview

The notebook demonstrates the complete pipeline for:

    Data preprocessing and cleaning

    Fine-tuning BERT-base-uncased for emotion classification

    Model evaluation and performance analysis

    Inference on custom text examples

    Model deployment and export

🏗️ Model Architecture

    Base Model: bert-base-uncased

    Task: Sequence classification (4 emotion classes)

    Training Framework: Hugging Face Transformers with PyTorch

    Training Hardware: GPU-accelerated (Tesla T4)

📊 Dataset

The model was trained on an emotion classification dataset with:

    Total samples: 22,050 text instances

    Classes: 4 emotions (Anger, Joy, Neutral, Sadness)

    Train/Validation/Test split: 17,860/1,985/2,205 samples

Data Distribution:

    Anger: ~26.5%

    Joy: ~26.4%

    Neutral: ~23.6%

    Sadness: ~23.4%

📈 Performance Metrics
Test Set Results:

    Accuracy: 65.7%

    Macro F1-Score: 66.0%

🚀 Usage
Installation
    
    pip install transformers datasets evaluate scikit-learn pandas matplotlib seaborn accelerate

Loading the Model
python

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained("bert_emotion_model")

    tokenizer = AutoTokenizer.from_pretrained("bert_emotion_model")

🛠️ Training Details
Hyperparameters:

    Learning Rate: 3e-5

    Batch Size: 16 (train), 32 (eval)

    Epochs: 3

    Max Sequence Length: 128 tokens

    Optimizer: AdamW with weight decay

📁 Project Structure
text

├── NLP_24f7807_Task1_P2(1).ipynb    # Main training notebook
├── bert_emotion_model/              # Saved model directory
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── vocab.txt
├── emotions-dataset.csv             # Raw dataset
├── emotion_labels.csv               # Label mappings

📊 Model Insights

    The model performs best on Anger detection (F1: 0.789)

    Neutral emotions are moderately well-classified (F1: 0.708)

    Joy and Sadness show room for improvement

    Confusion matrix reveals some confusion between Joy/Neutral and Sadness/Anger

Task2: Recipe Generation with GPT-2

This project implements a recipe generation system using GPT-2 fine-tuned on the 3A2M Extended Recipe dataset. The model can generate complete recipes (including ingredients and instructions) from dish names or ingredient prompts.
🎯 Project Overview

This implementation demonstrates:

    Fine-tuning GPT-2 on a large recipe dataset (3A2M Extended)

    Recipe generation with structured output (title, ingredients, instructions)

    Quality evaluation using ROUGE and BLEU metrics

    Interactive web interface using Gradio

📊 Dataset

The model is trained on the 3A2M Extended Recipe Dataset containing:

    2,231,143 recipe samples

    Structured format with:

        Recipe titles

        Ingredients (NER format)

        Cooking instructions

        Categories and labels

🚀 Features
1. Model Training

    Base Model: GPT-2

    Training Samples: 5,000 recipes (optimized subset)

    Epochs: 3

    Batch Size: 8

    GPU Optimized: Mixed precision training on CUDA

2. Recipe Generation

    Input: Dish name or ingredient list

    Output: Complete recipe with:

        Title

        Formatted ingredient list

        Numbered instructions

    Generation Parameters:

        Temperature: 0.9

        Top-p: 0.95

        Max length: 300 tokens

3. Quality Evaluation

Performance Metrics (on 5 test samples):

    ROUGE-1: 0.2908

    ROUGE-2: 0.0643

    ROUGE-L: 0.1538

    BLEU: 0.0991

📁 Project Structure
  recipe-generation-gpt2/
├── nlp_p2_Task2(2).ipynb          # Main training notebook
├── recipe-gpt2-5k/                # Trained model directory
├── recipe_gradio_app.py           # Gradio web interface
├── recipes_dataset/               # Dataset files
│   └── 3A2M_EXTENDED.csv

🛠️ Installation & Setup
Prerequisites
  bash
    pip install transformers datasets torch pandas numpy scikit-learn matplotlib seaborn tqdm gradio evaluate rouge-score nltk kagglehub
Dataset Download
  bash
    kaggle datasets download -d nazmussakibrupol/3a2mext
    unzip 3a2mext.zip -d recipes_dataset
🎮 Usage
1. Training the Model

Run the Jupyter notebook nlp_p2_Task2(2).ipynb to:

    Preprocess the dataset

    Fine-tune GPT-2

    Evaluate model performance

    Generate example recipes
📊 Evaluation Metrics

The model was evaluated using standard NLP metrics:

    ROUGE: Measures overlap between generated and reference text

    BLEU: Evaluates precision of n-gram matches

    Human Evaluation: Qualitative assessment of recipe coherence and practicality
🙏 Acknowledgments

    Hugging Face for the Transformers library

    Kaggle for hosting the 3A2M dataset

    Google Colab for GPU resources

Task3: Text Summarization with T5 Model

This project implements a text summarization system using the T5-small model, fine-tuned on the CNN/DailyMail dataset. The model can generate concise summaries from longer text articles and includes a user-friendly web interface for easy interaction

📋 Project Overview

    Model: T5-small transformer model fine-tuned for summarization

    Dataset: CNN/DailyMail newspaper articles (500 samples for training)

    Framework: Hugging Face Transformers

    Interface: Gradio web application

    Evaluation: Comprehensive metrics including ROUGE scores

🚀 Features

    Fast Training: Optimized for quick training with limited computational resources

    Web Interface: Easy-to-use Gradio interface for text summarization

    Comprehensive Evaluation: Multiple metrics including ROUGE-1, ROUGE-2, ROUGE-L, and BLEU scores

    Model Export: Complete model packaging for deployment

📊 Performance Metrics

The model achieves the following performance on the test set:

    ROUGE-1: 0.1883

    ROUGE-2: 0.0838

    ROUGE-L: 0.1713

    Keyword Overlap: 24.32%

    Coherence Rate: 100.00%

    Relevance Rate: 100.00%

🛠️ Installation & Setup
Prerequisites

    pip install transformers datasets pandas scikit-learn nltk gradio kagglehub evaluate rouge_score

Dataset Setup
bash

# Download and extract the CNN/DailyMail dataset
kaggle datasets download -d gowrishankarp/newspaper-text-summarization-cnn-dailymail
unzip newspaper-text-summarization-cnn-dailymail.zip -d cnn_dailymail_dataset

🏃‍♂️ Usage
Training the Model

The notebook includes a complete training pipeline:

    Data Loading: Loads 500 samples from the CNN/DailyMail dataset

    Preprocessing: Tokenizes articles and highlights for T5 model

    Training: Fine-tunes T5-small with optimized settings for CPU training

    Evaluation: Comprehensive performance analysis

Running the Web Interface
python

# Launch the Gradio interface
demo.launch(share=True)

The interface provides:

    Text input for articles

    Real-time summarization

    Example texts for quick testing

📁 Project Structure
text

├── cnn_dailymail_dataset/     # Dataset files
├── t5-tiny-final/            # Trained model and tokenizer
├── evaluation_report.json    # Performance metrics
└── t5_summarization_model.zip # Packaged model for deployment

🔧 Model Details

    Base Model: T5-small (60 million parameters)

    Training Data: 400 samples from CNN/DailyMail

    Test Data: 100 samples for evaluation

    Input Format: "summarize: [article text]"

    Output: Concise summary (max 50 tokens)

📈 Evaluation Results

The model demonstrates:

    Good entity recognition and coherent sentence structure

    Effective for short summaries with fast inference

    Practical performance suitable for educational and prototyping use

