# crime_mystery_gpt
This project try to explore the idea of the creation of the large language model for the crime mystery novels🧟🕷🔪👻🎬. It helps to understand the world of generative AI in a much better way. How can we do more in the space of large language model and maybe try to create a better story and narative to the readers.


Project Overview
This research project explores the application of a GPT-2 based language model for generating crime mystery stories. By leveraging deep learning techniques, we aim to understand the potential of transformer models in creating genre-specific narratives with intricate plot structures and character motivations.
Research Motivation
Crime and mystery genres present unique challenges for language models due to their complex narrative requirements:

Maintaining suspense
Preserving logical plot consistency
Capturing intricate character motivations
Gradual revelation of plot elements

Dataset
Data Source

Hugging Face Mystery Crime Books dataset by Aleksey Korshuk
Sourced from public domain texts, including works by Sir Arthur Conan Doyle

Dataset Statistics

Total characters: 562,489
Total tokens: 136,309
Train/Validation Split: 90/10

Model Architecture
Configuration

Model Type: GPT-2 Small
Total Parameters: 124 million
Vocabulary Size: 50,257 tokens
Context Length: 1,024 tokens
Embedding Dimension: 768
Transformer Layers: 12
Attention Heads: 12

Key Features

Token Embeddings
Positional Embeddings
Multi-head Self-Attention
Autoregressive Text Generation

Training Configuration

Optimizer: AdamW
Learning Rate: 0.0004
Weight Decay: 0.1
Dropout Rate: 0.1
Batch Size: 64
Total Epochs: 50
Hardware: Nvidia T4 GPU

Preprocessing Techniques

Regular expression-based tokenization
Byte Pair Encoding (BPE) for handling out-of-vocabulary words
Sliding window approach for sequence prediction

Evaluation Metrics

Cross-entropy loss
Perplexity
BLEU and ROUGE scores
Embedding-based similarity
Token-based F1 score for genre-specific elements

Key Challenges and Observations

Initial model generated nonsensical text
Observed overfitting due to limited dataset size
Difficulty in maintaining narrative consistency

Future Work

Expand dataset diversity
Implement advanced decoding strategies
Explore reinforcement learning techniques
Improve narrative coherence
Enhance genre-specific storytelling capabilities
