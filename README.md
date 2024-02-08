Data preparation process
========================

- Define the paths to the source (English) and target (Nepali) data files.
- Class **TranslationDataset** for creating custom dataset

### **TranslationDataset**
It reads the source and target data from the specified files and stores them in the src_data and trg_data attributes. The __len__ method returns the total number of samples in the dataset. The __getitem__ method retrieves a pair of source and target sentences at the given index. It lowercases the source sentence and removes leading/trailing whitespaces.

- The ShardingFilterIterDataPipe is used for shuffling and splitting the dataset into training, validation, and test sets.


The random_split method is called to split the dataset into training, validation, and test sets. The total_length parameter is set to the length of the original dataset. The weights parameter specifies the proportion of data to be allocated to each set. The seed parameter ensures reproducibility by providing a seed for the random number generator.

This process results in three subsets: train, val, and test, each containing a portion of the original dataset according to the specified weights. These subsets can then be used for training, validation, and testing your machine translation model.


Tokenization Process
=======================

For tokenizing source language(english): **torchtext** is used to obtain a tokenizer based on the specified library and language. In this case, it uses the 'spacy' tokenizer for English.

For tokenizing source language(Nepali): the indic_tokenize.trivial_tokenize function from **indicnlp library** is used. 

- *This, **indicnlp tokenizer** specifically designed for tokenizing Hindi text and since Hindi and Nepali have a same script and writing structure, this is used for tokenizing nepali text.*

Compared to other nepali/hindi tokenizer the **indicnlp** library tokenizer performs very well



**yield_tokens**: This is a helper function that takes a dataset (data) and a language (language) as input. It uses the previously defined token transforms (token_transform) to tokenize the sentences in the dataset based on the specified language.

The language_index dictionary is used to map language names to their corresponding indices in the data samples. For example, **SRC_LANGUAGE** is mapped to 0, and **TRG_LANGUAGE** is mapped to 1.

The function iterates over each data sample, retrieves the text based on the language index, and tokenizes the text using the appropriate tokenizer obtained earlier. It yields a list of tokens for each data sample.


Numericalization
=============

The numericalization process involves converting tokenized sentences into sequences of indices based on the vocabulary created. This step is crucial for feeding text data into neural networks or other machine learning models.

Here the vocab transform maps the token to indexes(numbers) and vice versa for source and target language respectiely.


Training with general attention
===================

BATCH_SIZE = 64
emb_dim     = 256  
hid_dim     = 512  
dropout     = 0.5

# Model architecture

### Seq2SeqPackedAttention

### Overview

Seq2SeqPackedAttention model with an encoder-decoder architecture.

#### Encoder

- **Embedding Layer:** Embedding with input size 7333 and output size 256.
- **RNN Layer:** Bidirectional GRU with input size 256 and hidden size 512.
- **Fully Connected Layer (fc):** Linear layer with in_features=1024 and out_features=512.
- **Dropout:** Dropout layer with a dropout probability of 0.5.

#### Decoder

- **Attention Layer:**
  - **v Layer:** Linear layer with in_features=512 and out_features=1.
  - **U Layer:** Linear layer with in_features=1024 and out_features=512.
- **Embedding Layer:** Embedding with input size 18612 and output size 256.
- **RNN Layer:** GRU with input size 1280 and hidden size 512.
- **Fully Connected Layer (fc):** Linear layer with in_features=1792 and out_features=18612.
- **Dropout:** Dropout layer with a dropout probability of 0.5.


Total parameters of model with general attention is **46184372**

| Test Loss: 6.582 | Test PPL: 722.102 |



Attention Map
===================
![alt text](./app/static/attention.png?raw=true)



### Web app Documentation

##### **Overview**
This is flask web application allows users to input a text in source language ie. **English** text area and translates the source language into targe language ie **Nepali**.

- This web application consists of two web pages - Home Page(*index.html*) and Result Page(*result.html*)
   * Home Page: 
   ![alt text](./app/staticindex.png?raw=true)
   Here user can input text in source language to the text area and result translated text in traget language is shown in result section.

  


##### **Language Translation**
On providing the text in source language in home page to translate the text, **translate_eng_to_nepali** function from the **translate.py** file.

- **translate** method is responsible for translating text for source language(english) to target language (nepali).

##### **Running the Application**
The Flask application is run using python app.py in the terminal, and the web interface can be accessed at http://127.0.0.1:5000/.

