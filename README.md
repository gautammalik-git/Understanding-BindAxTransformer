# Understanding-BindAxTransformer
If you're interested in developing a machine learning model to learn the intricate interactions between proteins and ligands, this repository is for you. Here, I provide a detailed, line-by-line walkthrough of the BindAxTransformer code, explaining how transformers operate and how they can be applied in drug discovery.

# BindAxTransformer: A Transformer Model for Ligand-Protein Interactions

## Overview

Welcome to the BindAxTransformer repository! This project features a self-supervised transformer model designed to understand and predict interactions between ligands and proteins. The model is built using PyTorch and Hugging Face's Transformers library.

In this README, we provide a detailed walkthrough of the code, explaining each section and its purpose. This guide is ideal for those interested in learning how transformers work in the context of protein-ligand interactions.

## Code Explanation

### Importing Libraries

```python
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
```

`import os` : Handles file operations such as listing files in a directory.
<br>
`import numpy as np`: Performs numerical operations, such as distance calculations.
<br>
`import torch`: Utilizes PyTorch for building and training neural networks.
<br>
`import torch.nn as nn`: Defines and uses neural network layers.
<br>
`from torch.utils.data import Dataset, DataLoader`: Manages and batches the dataset.
<br>
`from transformers import BertConfig, BertModel`: Utilizes BERT from Hugging Face for the transformer model.
<br>
`from sklearn.model_selection import train_test_split`: Splits the dataset into training and validation sets.
<br>
`from torch.nn.utils.rnn import pad_sequence`: Pads sequences for batch processing.
<br>

### Tokenizing Ligand Atoms

```python
def tokenize_ligand(atom_lines):
    ligand_tokens = []
    for line in atom_lines:
        if line.startswith('HETATM'):
            parts = line.split()
            if len(parts) >= 12:
                atom_type = parts[11].strip()
                try:
                    x, y, z = map(float, parts[6:9])
                    ligand_tokens.append({
                        'atom_type': atom_type,
                        'x': x, 'y': y, 'z': z,
                    })
                except ValueError:
                    continue
    return ligand_tokens
```
`def tokenize_ligand(atom_lines)`:: Defines a function to extract ligand atom information from PDB lines.
<br>
`ligand_tokens = []`: Initializes a list for storing ligand tokens.
<br>
`for line in atom_lines:`: Iterates over each line of the atom data.
<br>
`if line.startswith('HETATM'):`: Filters lines representing ligand atoms.
<br>
`parts = line.split()`: Splits the line into components.
<br>
`if len(parts) >= 12:`: Checks for sufficient data in the line.
<br>
`atom_type = parts[11].strip()`: Extracts the atom type.
<br>
`try: x, y, z = map(float, parts[6:9])`: Converts coordinates to floats. If successful:
<br>
`ligand_tokens.append({...})`: Adds the token information to the list.
<br>
`except ValueError: continue`: Skips lines with invalid data.
<br>
`return ligand_tokens`: Returns the list of ligand tokens.
<br>

### Tokenizing Protein Binding Sites

```python
def tokenize_protein_binding_site(atom_lines, ligand_tokens, distance_threshold=5.0):
    binding_site_tokens = []
    
    def calculate_distance(coord1, coord2):
        return np.linalg.norm(np.array(coord1) - np.array(coord2))
    
    for line in atom_lines:
        if line.startswith('ATOM'):
            parts = line.split()
            if len(parts) >= 12:
                residue_type = parts[3].strip()
                atom_type = parts[11].strip()
                try:
                    x, y, z = map(float, parts[6:9])
                    protein_coord = (x, y, z)
                    for ligand in ligand_tokens:
                        ligand_coord = (ligand['x'], ligand['y'], ligand['z'])
                        if calculate_distance(protein_coord, ligand_coord) <= distance_threshold:
                            binding_site_tokens.append({
                                'residue_type': residue_type,
                                'atom_type': atom_type,
                                'x': x, 'y': y, 'z': z
                            })
                            break
                except ValueError:
                    continue
    return binding_site_tokens
```

`def tokenize_protein_binding_site(atom_lines, ligand_tokens, distance_threshold=5.0):`: Defines a function to tokenize protein binding sites.
<br>
`binding_site_tokens = []`: Initializes a list for storing binding site tokens.
<br>
`def calculate_distance(coord1, coord2):`: Defines a helper function to compute Euclidean distance.
<br>
`return np.linalg.norm(np.array(coord1) - np.array(coord2))`: Computes and returns the distance.
<br>
`for line in atom_lines:`: Iterates over each line of protein data.
<br>
`if line.startswith('ATOM'):`: Filters lines representing protein atoms.
<br>
`parts = line.split()`: Splits the line into components.
<br>
`if len(parts) >= 12:`: Checks for sufficient data in the line.
<br>
`residue_type = parts[3].strip()`: Extracts the residue type.
<br>
`atom_type = parts[11].strip()`: Extracts the atom type.
<br>
`try: x, y, z = map(float, parts[6:9])`: Converts coordinates to floats. If successful:
<br>
`protein_coord = (x, y, z)`: Creates a tuple for protein coordinates.
<br>
`for ligand in ligand_tokens:`: Iterates through the ligand tokens.
<br>
`ligand_coord = (ligand['x'], ligand['y'], ligand['z'])`: Creates a tuple for ligand coordinates.
<br>
`if calculate_distance(protein_coord, ligand_coord) <= distance_threshold:`: Checks if the distance is within the threshold.
<br>
`binding_site_tokens.append({...})`: Adds the protein atom to the list if within the distance threshold.
<br>
`except ValueError: continue`: Skips lines with invalid data.
<br>
`return binding_site_tokens`: Returns the list of binding site tokens.
<br>

### Parsing and Tokenizing PDB Files

```python
def parse_and_tokenize_pdb(pdb_file, distance_threshold=5.0):
    with open(pdb_file, 'r') as file:
        lines = file.readlines()
    
    ligand_lines = [line for line in lines if line.startswith('HETATM')]
    protein_lines = [line for line in lines if line.startswith('ATOM')]
    
    ligand_tokens = tokenize_ligand(ligand_lines)
    binding_site_tokens = tokenize_protein_binding_site(protein_lines, ligand_tokens, distance_threshold)
    
    return ligand_tokens, binding_site_tokens
```
<br>

`def parse_and_tokenize_pdb(pdb_file, distance_threshold=5.0):`: Defines a function to parse and tokenize a PDB file.
<br>
`with open(pdb_file, 'r') as file:`: Opens the PDB file for reading.
<br>
`lines = file.readlines()`: Reads all lines from the file.
<br>
`ligand_lines = [line for line in lines if line.startswith('HETATM')]`: Extracts lines for ligands.
<br>
`protein_lines = [line for line in lines if line.startswith('ATOM')]`: Extracts lines for proteins.
<br>
`ligand_tokens = tokenize_ligand(ligand_lines)`: Tokenizes the ligand lines.
<br>
`binding_site_tokens = tokenize_protein_binding_site(protein_lines, ligand_tokens, distance_threshold)`: Tokenizes the binding site lines.
<br>
`return ligand_tokens, binding_site_tokens`: Returns the tokenized data.
<br>

### Combining Tokens

```python
def combine_ligand_protein_tokens(ligand_tokens, binding_site_tokens):
    combined_tokens = [token['atom_type'] for token in ligand_tokens] + [token['residue_type'] for token in binding_site_tokens]
    segment_ids = [0] * len(ligand_tokens) + [1] * len(binding_site_tokens)
    return combined_tokens, segment_ids
```
<br>

`def combine_ligand_protein_tokens(ligand_tokens, binding_site_tokens):`: Defines a function to combine ligand and binding site tokens.
<br>
`combined_tokens = [token['atom_type'] for token in ligand_tokens] + [token['residue_type'] for token in binding_site_tokens]`: Merges token types from both ligand and binding site.
<br>
`segment_ids = [0] * len(ligand_tokens) + [1] * len(binding_site_tokens)`: Assigns segment IDs to differentiate between ligand and binding site tokens.
<br>
`return combined_tokens, segment_ids`: Returns the combined tokens and segment IDs.

### Processing PDB Files

### Code Explanation

#### Processing PDB Files

```python
def process_pdb_files(directory):
    all_combined_tokens = []
    all_segment_ids = []
    
    for pdb_file in os.listdir(directory):
        if pdb_file.endswith('.pdb'):
            file_path = os.path.join(directory, pdb_file)
            ligand_tokens, binding_site_tokens = parse_and_tokenize_pdb(file_path)
            combined_tokens, segment_ids = combine_ligand_protein_tokens(ligand_tokens, binding_site_tokens)
            all_combined_tokens.append(combined_tokens)
            all_segment_ids.append(segment_ids)
    
    return all_combined_tokens, all_segment_ids
```
<br>

`def process_pdb_files(directory):`: Defines a function to process PDB files in a given directory.
<br>
`all_combined_tokens = []`: Initializes a list to store tokens from all processed PDB files.
<br>
`all_segment_ids = []`: Initializes a list to store segment IDs from all processed PDB files.
<br>
`for pdb_file in os.listdir(directory):`: Iterates over each file in the specified directory.
<br>
`if pdb_file.endswith('.pdb'):`: Checks if the file has a .pdb extension.
<br>
`file_path = os.path.join(directory, pdb_file)`: Constructs the full path to the PDB file.
<br>
`ligand_tokens, binding_site_tokens = parse_and_tokenize_pdb(file_path)`: Parses and tokenizes the PDB file to extract ligand and binding site tokens.
<br>
`combined_tokens, segment_ids = combine_ligand_protein_tokens(ligand_tokens, binding_site_tokens)`: Combines and segments tokens from ligand and binding site.
<br>
`all_combined_tokens.append(combined_tokens)`: Adds the combined tokens to the list.
<br>
`all_segment_ids.append(segment_ids)`: Adds the segment IDs to the list.
<br>
`return all_combined_tokens, all_segment_ids`: Returns the lists of combined tokens and segment IDs.
<br>

### Creating the Dataset Class

```python
class ProteinLigandDataset(Dataset):
    def __init__(self, input_ids, segment_ids):
        self.input_ids = input_ids
        self.segment_ids = segment_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'token_type_ids': torch.tensor(self.segment_ids[idx], dtype=torch.long),
            'attention_mask': torch.ones(len(self.input_ids[idx]), dtype=torch.long)
        }
```
<br>

`class ProteinLigandDataset(Dataset):`: Defines a custom dataset class for handling protein-ligand interaction data.
<br>
`def __init__(self, input_ids, segment_ids):`: Initializes the dataset with input IDs and segment IDs.
<br>
`self.input_ids = input_ids`: Stores input IDs as an instance variable.
<br>
`self.segment_ids = segment_ids`: Stores segment IDs as an instance variable.
<br>
`def __len__(self):`: Returns the number of samples in the dataset.
<br>
`return len(self.input_ids)`: Returns the length of the input IDs list.
<br>
`def __getitem__(self, idx):`: Retrieves an item from the dataset at a specified index.
<br>
`return { ... }`: Returns a dictionary containing tensors for input IDs, segment IDs, and attention masks.
<br>

### Colate Function

```python
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }
```
<br>

`def collate_fn(batch):`: Defines a function to collate data into batches for training.
<br>
`input_ids = [item['input_ids'] for item in batch]`: Extracts input IDs from each item in the batch.
<br>
`token_type_ids = [item['token_type_ids'] for item in batch]`: Extracts token type IDs from each item in the batch.
<br>
`attention_mask = [item['attention_mask'] for item in batch]`: Extracts attention masks from each item in the batch.
<br>
`input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)`: Pads input IDs to the same length.
<br>
`token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)`: Pads token type IDs to the same length.
<br>
`attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)`: Pads attention masks to the same length.
<br>
`return { ... }`: Returns a dictionary with padded sequences.
<br>

### Defining the Transformer Model

```python
class ProteinLigandTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_attention_heads=12, num_hidden_layers=12):
        super(ProteinLigandTransformer, self).__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=2
        )
        self.bert = BertModel(config)
        self.cls = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.cls(sequence_output)
        return logits
```
<br>

`class ProteinLigandTransformer(nn.Module):`: Defines a transformer model for protein-ligand interactions.
<br>
`def __init__(self, vocab_size, hidden_size=768, num_attention_heads=12, num_hidden_layers=12):`: Initializes the transformer with configuration parameters.
<br>
`super(ProteinLigandTransformer, self).__init__()`: Calls the parent class constructor.
<br>
`config = BertConfig(...)`: Sets up the BERT configuration with specified parameters.
<br>
`self.bert = BertModel(config)`: Initializes the BERT model with the configuration.
<br>
`self.cls = nn.Linear(hidden_size, vocab_size)`: Defines a linear layer for classification.
<br>
`def forward(self, input_ids, token_type_ids, attention_mask=None):`: Defines the forward pass of the model.
<br>
`outputs = self.bert(...)`: Passes inputs through the BERT model.
<br>
`sequence_output = outputs[0]`: Extracts the sequence output from BERT.
<br>
`logits = self.cls(sequence_output)`: Applies the linear layer to the sequence output.
<br>
`return logits`: Returns the logits
<br>

### Main training loop 

```python
def main():
    directory = 'nrsites'
    all_combined_tokens, all_segment_ids = process_pdb_files(directory)

    unique_tokens = list(set([token for tokens in all_combined_tokens for token in tokens]))
    token2id = {token: idx for idx, token in enumerate(unique_tokens)}
    id2token = {idx: token for token, idx in token2id.items()}

    input_ids = [[token2id[token] for token in tokens] for tokens in all_combined_tokens]

    train_input_ids, val_input_ids, train_segment_ids, val_segment_ids = train_test_split(
        input_ids, all_segment_ids, test_size=0.2, random_state=42
    )

    train_dataset = ProteinLigandDataset(input_ids=train_input_ids, segment_ids=train_segment_ids)
    val_dataset = ProteinLigandDataset(input_ids=val_input_ids, segment_ids=val_segment_ids)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinLigandTransformer(vocab_size=len(token2id)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            
            labels = input_ids.clone()
            mask = torch.bernoulli(torch.full(input_ids.shape, 0.15)).bool()
            labels[~mask] = -100  # Only compute loss on masked tokens

            loss = criterion(outputs.view(-1, len(token2id)), labels.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                
                labels = input_ids.clone()
                mask = torch.bernoulli(torch.full(input_ids.shape, 0.15)).bool()
                labels[~mask] = -100

                loss = criterion(outputs.view(-1, len(token2id)), labels.view(-1))
                val_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}, Val Loss: {val_loss / len(val_dataloader):.4f}')

    torch.save(model.state_dict(), 'protein_ligand_model.pth')

if __name__ == "__main__":
    main()
```
<br>

`def main():`: Defines the main function to execute the training process.
<br>
`directory = '<name>'`: Specifies the directory containing PDB files.
<br>
`all_combined_tokens, all_segment_ids = process_pdb_files(directory)`: Processes PDB files to get tokens and segment IDs.
<br>
`unique_tokens = list(set([token for tokens in all_combined_tokens for token in tokens]))`: Extracts unique tokens from all combined tokens.
<br>
`token2id = {token: idx for idx, token in enumerate(unique_tokens)}`: Maps tokens to unique IDs.
<br>
`id2token = {idx: token for token, idx in token2id.items()}`: Maps IDs back to tokens.
<br>
`input_ids = [[token2id[token] for token in tokens] for tokens in all_combined_tokens]`: Converts tokens to their corresponding IDs.
<br>
`train_input_ids, val_input_ids, train_segment_ids, val_segment_ids = train_test_split(...)`: Splits the data into training and validation sets.
<br>
`train_dataset = ProteinLigandDataset(input_ids=train_input_ids, segment_ids=train_segment_ids)`: Creates a dataset for training.
<br>
`val_dataset = ProteinLigandDataset(input_ids=val_input_ids, segment_ids=val_segment_ids)`: Creates a dataset for validation.
<br>
`train_dataloader = DataLoader(...)`: Sets up a DataLoader for training.
<br>
`val_dataloader = DataLoader(...)`: Sets up a DataLoader for validation.
<br>
`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`: Selects the device for training (GPU or CPU).
<br>
`model = ProteinLigandTransformer(vocab_size=len(token2id)).to(device)`: Initializes and moves the model to the selected device.
<br>
`optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)`: Sets up the optimizer for training.
<br>
`criterion = nn.CrossEntropyLoss(ignore_index=-100)`: Defines the loss function.
<br>
`num_epochs = 10`: Specifies the number of training epochs.
<br>
`for epoch in range(num_epochs):`: Loops over each epoch for training.
<br>
`model.train()`: Sets the model to training mode.
<br>
`train_loss = 0`: Initializes the training loss for the epoch.
<br>
`for batch in train_dataloader:`: Iterates over batches of training data.
<br>
`input_ids = batch['input_ids'].to(device)`: Moves input IDs to the selected device.
<br>
`token_type_ids = batch['token_type_ids'].to(device)`: Moves token type IDs to the selected device.
<br>
`attention_mask = batch['attention_mask'].to(device)`: Moves attention masks to the selected device.
<br>
`optimizer.zero_grad()`: Clears old gradients.
<br>
`outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)`: Computes model outputs.
<br>
`labels = input_ids.clone()`: Creates labels for loss computation.
<br>
`mask = torch.bernoulli(torch.full(input_ids.shape, 0.15)).bool()`: Generates a mask for token prediction.
<br>
`labels[~mask] = -100`: Sets masked tokens to -100 (ignored in loss computation).
<br>
`loss = criterion(outputs.view(-1, len(token2id)), labels.view(-1))`: Computes the loss.
<br>
`loss.backward()`: Computes gradients.
<br>
`optimizer.step()`: Updates model parameters.
<br>
`train_loss += loss.item()`: Accumulates training loss.
<br>
`model.eval()`: Sets the model to evaluation mode.
<br>
`val_loss = 0`: Initializes the validation loss.
<br>
`with torch.no_grad():`: Disables gradient computation for validation.
<br>
`for batch in val_dataloader:`: Iterates over batches of validation data.
<br>
`outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)`: Computes model outputs for validation.
<br>
`loss = criterion(outputs.view(-1, len(token2id)), labels.view(-1))`: Computes the validation loss.
<br>
`val_loss += loss.item()`: Accumulates validation loss.
<br>
`print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss`: {train_loss / len(train_dataloader):.4f}, Val Loss: {val_loss / len(val_dataloader):.4f}'): Prints the training and validation loss for each epoch.
<br>
`torch.save(model.state_dict(), 'protein_ligand_model.pth')`: Saves the trained model.
<br>
`if __name__ == "__main__":`: Ensures that main() runs only when the script is executed directly.
<br>
`main()`: Calls the main() function to start the training process.
<br>

