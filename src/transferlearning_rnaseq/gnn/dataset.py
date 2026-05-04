import os
import torch
from torch_geometric.data import InMemoryDataset, Dataset, Data
import polars as pl
class PPIDataset(Dataset):
    def __init__(
        self, root, result, ppi, protein_to_gene_mapping,
        exclude_columns, label_column='tumor_class',
        dtype=torch.float32, threshold=850,  transform=None, pre_transform=None, pre_filter=None
    ):
        self.result = result
        self.ppi = ppi
        self.protein_to_gene_mapping = protein_to_gene_mapping
        self.exclude_columns = exclude_columns
        self.label_column = label_column
        self.labels = result.select(label_column)
        self.dtype = dtype
        self.threshold = threshold
        super().__init__(root, transform, pre_transform, pre_filter)

        # Ici, on génère les fichiers si ce n'est pas déjà fait
        if not all([os.path.exists(os.path.join(self.processed_dir, f'data_{i}.pt')) for i in range(len(self.result))]):
            self.process()

    @property
    def raw_file_names(self):
        return ['dummy.txt']

    @property
    def processed_file_names(self):
        # On sauvegarde un fichier par échantillon
        return [f'data_{i}.pt' for i in range(len(self.result))]

    def process(self):
        protein_to_gene_dict = {
            row["Protein stable ID"]: row["Gene stable ID"]
            for row in self.protein_to_gene_mapping.iter_rows(named=True)
        }
        filtered_ppi = self.ppi.filter(pl.col("combined_score") > self.threshold)
        unique_genes = set(protein_to_gene_dict.values())
        gene_to_index = {gene: idx for idx, gene in enumerate(unique_genes)}

        edge_index = []
        edge_attr = []
        for r in filtered_ppi.iter_rows(named=True):
            protein1 = r["protein1_preprocess"]
            protein2 = r["protein2_preprocess"]
            if protein1 in protein_to_gene_dict and protein2 in protein_to_gene_dict:
                gene1 = protein_to_gene_dict[protein1]
                gene2 = protein_to_gene_dict[protein2]
                if gene1 in gene_to_index and gene2 in gene_to_index:
                    edge_index.append([gene_to_index[gene1], gene_to_index[gene2]])
                    edge_attr.append(r["combined_score"])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)

        for idx in range(len(self.result)):
            row = self.result[idx]
            node_features = row.select(pl.exclude(self.exclude_columns)).to_numpy()
            x = torch.tensor(node_features, dtype=torch.float32).view(-1, 1)
            if isinstance(self.label_column, str): y = torch.tensor(row[self.label_column], dtype=self.dtype)
            else: 
                label_values = self.labels.row(idx)
                y = {
                col: torch.tensor(val, dtype=self.dtype.get(col, torch.float32))
                for col, val in zip(self.labels.columns, label_values)
            }
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y
            )
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.result)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)
        return data