# Data

Below are descriptions of files which exist in our forager package. Primarily, the fluency list and lexical data are used in the pipeline

### Fluency Lists
The fluency list folder contains existing fluency list data that can be utilized with our pipeline. We provide two initial datasets, including the *"psych review"* dataset, which is the original dataset used in analysis in Hills TT, Jones MN, Todd PM (2012). The second dataset, is a clinical dataset from a Dartmouth Study. 

Fluency Lists should be formatted as follows:
- There are no headers in the data files
- There are two columns. The first column is the Subject ID. The second column is a fluency list item
- Each row represents a singular fluency list item from a fluency list
- Columns are delimited by a tabspace ('\t')

You are welcome to place your own fluency data in this folder, as long as they follow the specified format.

### Lexical Data
The lexical data folder contains multiple forms of lexical data. 

These include the following:
- vocab.csv : The vocabulary (animals), including an `UNK` entry used for out-of-vocabulary items when the mean-vector option is selected.
- USE_frequencies.csv : Word frequencies (Zipf frequencies from the `wordfreq` package) for each vocabulary item.
- USE_embeddings.csv : Universal Sentence Encoder embeddings for each vocabulary item, from which the semantic similarity matrix is calculated.
- USE_semantic_matrix.csv : An N by N matrix of semantic (cosine) similarity between vocabulary items.
- USE_phonological_matrix.csv : An N by N matrix of phonological similarity between vocabulary items.

These files are copies of the forager package's `data/lexical_data/animals/` folder.

You can provide your own lexical data too, which we provide functions for. However, it should follow similar format to the outline data. 

### Misc 
The misc folder includes miscellaneous data needed for any testing, methods, etc. For the most part, this folder is not used. 

### Norms
The norms folder includes any categorization-based norms data, such as the Troyer Norms (which we include by default). This is where you can place any categorization data that you would like to perform analysis with.
