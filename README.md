# Understanding-BindAxTransformer
If you're interested in developing a machine learning model to learn the intricate interactions between proteins and ligands, this repository is for you. Here, I provide a detailed, line-by-line walkthrough of the BindAxTransformer code, explaining how transformers operate and how they can be applied in drug discovery.

This code implements a machine learning model using transformer architecture (BERT) to analyze protein-ligand interactions from PDB files. I'll walk through it line by line, explaining the operations and math behind the model.

### **1. Imports and Setup**

```python

import os

import numpy as np

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from transformers import BertConfig, BertModel

from sklearn.model_selection import train_test_split

```

Here, necessary libraries are imported:

- `os` handles file and directory operations.

- `numpy` (`np`) is used for mathematical operations, particularly for handling coordinates.

- `torch` and `torch.nn` are used for constructing and training the deep learning model in PyTorch.

- `BertConfig` and `BertModel` are from the Hugging Face `transformers` library, which provides the pre-trained BERT model and its configuration.

- `train_test_split` from `sklearn` splits the dataset for training and validation.

### **2. Atom Tokenization**

```python

def tokenize_atom(line, molecule_type):

    ...

    return {

        'molecule_type': molecule_type,

        ...

    }

```

- **Goal**: This function extracts specific information from lines in PDB files and tokenizes atoms, separating ligand from protein atoms.

  - **Input**: A single line from a PDB file and a `molecule_type` (0 for ligand, 1 for protein).

  - **Tokenization**:

    - Extract atom type (columns 12--16), element (columns 76--78), and 3D coordinates (columns 30--54).

    - For protein atoms, it also gets residue name (columns 17--20) and residue number (columns 22--26).

  - **Output**: A dictionary with the atom's properties: type, element, coordinates, etc.

### **3. Parsing and Tokenizing PDB Files**

```python

def parse_and_tokenize_pdb(pdb_file):

    ...

    return ligand_tokens + protein_tokens

```

- **Goal**: Read a PDB file and separate ligand (`HETATM`) and protein (`ATOM`) atoms.

  - **Process**: The file is read line by line, and the `tokenize_atom()` function is applied based on the atom type.

  - **Output**: Returns a combined list of ligand and protein tokens.

### **4. Calculate Distance Between Atoms**

```python

def calculate_distance(coord1, coord2):

    return np.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

```

- **Mathematics**: This function calculates the Euclidean distance between two 3D points (atom coordinates) using the formula:

```math

  d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2 + (z_1 - z_2)^2}
```

### **5. Finding Protein-Ligand Interactions**

```python

def find_interactions(ligand_tokens, protein_tokens, distance_threshold=5.0):

    ...

    return interactions

```

- **Goal**: Find close protein-ligand atom pairs where the distance between them is less than a specified threshold (5 Å).

  - **Process**: The function loops through every pair of ligand and protein atoms, calculates their distance, and stores pairs that interact (distance ≤ 5.0).

### **6. Processing Multiple PDB Files**

```python

def process_pdb_files(directory):

    ...

    return all_tokens

```

- **Goal**: Process all `.pdb` files in a directory.

  - **Process**:

    1. For each file, it parses and tokenizes atoms.

    2. Calls `find_interactions()` to get all interacting atoms.

    3. Stores the tokenized information for further use.

### **7. Protein-Ligand Dataset Class**

```python

class ProteinLigandDataset(Dataset):

    ...

    def __getitem__(self, idx):

        ...

```

- **Goal**: Define a custom dataset class for PyTorch.

  - **Methods**:

    - `__len__()`: Returns the length of the dataset.

    - `__getitem__()`: Fetches a single data point (input IDs, coordinates, distance).

    - The model uses tokenized IDs for ligand and protein atom types and their coordinates.

### **8. Collate Function**

```python

def collate_fn(batch):

    ...

    return {

        ...

    }

```

- **Goal**: Prepare batches of data during training.

  - **Process**:

    1. The function pads sequences (`input_ids`) for uniform length across the batch.

    2. Coordinates and distances are stacked into tensors.

    3. An attention mask is also created to indicate the positions of valid tokens.

### **9. Protein-Ligand Transformer Model**

```python

class ProteinLigandTransformer(nn.Module):

    ...

    def forward(self, input_ids=None, coordinates=None, distances=None, attention_mask=None):

        ...

        return logits

```

- **Architecture**:

  - **BERT Initialization**: 

```python

    config = BertConfig(...), self.bert = BertModel(config)

```

    This initializes the transformer with:

    - `hidden_size=1152`: The size of hidden representations for each token.

    - `num_attention_heads=12`: The number of heads in the multi-head attention mechanism.

    - `num_hidden_layers=12`: The number of transformer layers (each with self-attention and feed-forward networks).

  - **Encoding Coordinates and Distances**:

```python

    self.coordinate_encoder = nn.Linear(6, hidden_size)

    self.distance_encoder = nn.Linear(1, hidden_size)

```

    - The model encodes both the 3D coordinates of the ligand and protein atoms (6 inputs for x, y, z of both molecules) and the distance (1 input) into the same dimension (`hidden_size`).

    - These embeddings are combined with the token embeddings from BERT.

  - **Forward Method**:

    1. **Token Embeddings**: 

```python

       token_embeddings = self.bert.embeddings(input_ids)

```

       This creates embeddings for the ligand and protein atom types.

    2. **Adding Coordinate and Distance Embeddings**:

```python

       embeddings = token_embeddings + coordinate_embeddings + distance_embeddings

```

       Here, token embeddings are combined with coordinate and distance embeddings to encode spatial and interaction information.

    3. **Final Output**: 

```python

       logits = self.cls(sequence_output)

```

       The model's output is passed through a classification layer to predict the next tokens.

### **10. Main Function**

```python

def main():

    directory = 'mysite'

    ...

    torch.save(model.state_dict(), 'protein_ligand_model.pth')

```

- **Process**:

  1. The PDB files are processed to extract all tokenized data.

  2. The data is split into training and validation sets.

  3. A vocabulary (`token2id`) is created for mapping tokens to IDs.

  4. The model is trained using the masked language model (MLM) objective:

     - **Masking tokens**: 15% of tokens are randomly masked, and the model predicts the masked tokens.

     - **Loss**: Cross-entropy loss is used to train the model to correctly predict the masked tokens.

  5. **Model Saving**: The trained model is saved for later use.

---

### **Mathematics of BERT Model**

The transformer model uses **self-attention** to compute relationships between tokens, regardless of their positions. The key mathematical operations in self-attention include:

1\. **Self-attention**:

  $$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V$$

   - `Q`, `K`, `V` are query, key, and value matrices derived from the input.

   - \( d_k \) is the dimension of the keys.

2\. **Multi-head attention**: This involves running multiple attention heads in parallel and concatenating their outputs.

3\. **Feed-forward layers**: Each token representation is passed through a fully connected network after the self-attention step.

This process is repeated for each layer in the transformer, ultimately producing embeddings that consider both the spatial (3D coordinates) and interaction (distances) contexts of ligand-protein complexes.
