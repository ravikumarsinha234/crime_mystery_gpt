# crime_mystery_gpt
This project try to explore the idea of the creation of the large language model for the crime mystery novels🧟🕷🔪👻🎬. It helps to understand the world of generative AI in a much better way. How can we do more in the space of large language model and maybe try to create a better story and narative to the readers.


# Crime Mystery Story Generation using GPT-2

## Overview
A research project exploring the use of GPT-2 to generate crime mystery stories, focusing on understanding the model's ability to capture genre-specific narrative elements.

## Dataset Details
- **Source**: Hugging Face Mystery Crime Books dataset
- **Total Characters**: 562,489
- **Total Tokens**: 136,309
- **Train/Validation Split**: 90/10
- **Text Sources**: Public domain crime mystery novels, including works by Sir Arthur Conan Doyle

## Model Specifications
- **Model Type**: GPT-2 Small
- **Total Parameters**: 124 million
- **Vocabulary Size**: 50,257 tokens
- **Context Length**: 1,024 tokens
- **Embedding Dimension**: 768
- **Transformer Layers**: 12
- **Attention Heads**: 12

## Training Setup
- **Optimizer**: AdamW
- **Learning Rate**: 0.0004
- **Weight Decay**: 0.1
- **Dropout Rate**: 0.1
- **Batch Size**: 64
- **Total Epochs**: 50
- **Hardware**: Nvidia T4 GPU

## Research Challenges
- Generating coherent crime mystery narratives
- Maintaining suspense and plot consistency
- Handling complex character motivations
- Overcoming limited dataset diversity


## Future Research Directions
1. Expand dataset to include more diverse crime mystery texts
2. Implement advanced decoding strategies
3. Explore reinforcement learning techniques
4. Improve narrative coherence
5. Enhance genre-specific storytelling capabilities


## Reference
The work we performed is inspired from the sebastian raschka and andrej karapathy. Due credit should go to them.

## References

1. **Raschka, S.** (2024). *Build A Large Language Model (From Scratch)*. Manning. ISBN: 978-1633437166.  
   Available at: [Manning](https://www.manning.com/books/build-a-large-language-model-from-scratch)  
   GitHub Repository: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
2. https://www.youtube.com/watch?v=l8pRSuU81PU

3. Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Pushmeet Kohli, and James Allen. A corpus and cloze evaluation for deeper understanding of commonsense stories. In Kevin Knight, Ani Nenkova, and Owen Rambow, editors, Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 839–849, San Diego, California, June 2016. Association for Computational Linguistics. 

4. Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277, 2023. 

5. Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. 

6. Ashish Vaswani, Noam Shazeer, Niki Parmar, et al. Attention is all you need. Advances in neural information processing systems, 30, 2017. 

7. Rui Xu, Xintao Wang, Jiangjie Chen, Siyu Yuan, Xinfeng Yuan, Jiaqing Liang, Zulong Chen, Xiaoqing Dong, and Yanghua Xiao. Character is destiny: Can role-playing language agents make persona-driven decisions?, 2024. 

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/crime-mystery-gpt.git

## Citation
If you use this work in your research, please cite:

Sinha, R. K., & Esaulov, V. (2022). Building a GPT for Character-Level Language Modeling.



