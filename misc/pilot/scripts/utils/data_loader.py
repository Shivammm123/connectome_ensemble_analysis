"""
Data loading utilities with automatic neuron name mapping.
Replicates Liessem et al. (2025) methodology.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml


class ConnectomeDataLoader:
    """Load and process FlyWire BANC connectome data with name mapping."""

    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
        self.connections = None
        self.neurons = None
        self.name_mapping = None
        self._name_lookup: Dict[int, str] = {}

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_all_data(self, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data files.

        Returns
        -------
        connections : pd.DataFrame
        neurons : pd.DataFrame
        name_mapping : pd.DataFrame
        """
        if verbose:
            print("Loading FlyWire BANC data...")

        self.connections = self.load_connections(verbose=verbose)
        self.neurons = self.load_neurons(verbose=verbose)
        self.name_mapping = self.create_name_mapping(verbose=verbose)

        return self.connections, self.neurons, self.name_mapping

    def load_connections(self, verbose: bool = True) -> pd.DataFrame:
        """Load connectivity data with standardized column names."""
        conn_file = Path(self.config['data']['connections'])

        if not conn_file.exists():
            raise FileNotFoundError(
                f"Connections file not found: {conn_file}\n"
                f"Please place connections_princeton.csv in data/raw/"
            )

        if verbose:
            print(f"  Loading {conn_file.name}...")

        conn = pd.read_csv(conn_file)
        conn = conn.rename(columns={
            'pre_root_id': 'source',
            'post_root_id': 'target',
            'syn_count': 'weight'
        })

        essential_cols = ['source', 'target', 'weight']
        optional_cols = ['neuropil', 'nt_type']
        cols_to_keep = essential_cols + [c for c in optional_cols if c in conn.columns]
        conn = conn[cols_to_keep]

        if verbose:
            print(f"    ✓ {len(conn):,} connections")
            print(f"    ✓ {conn['source'].nunique():,} source neurons")
            print(f"    ✓ {conn['target'].nunique():,} target neurons")

        return conn

    def load_neurons(self, verbose: bool = True) -> pd.DataFrame:
        """Load neuron attributes."""
        neurons_file = Path(self.config['data']['neurons'])

        if not neurons_file.exists():
            raise FileNotFoundError(
                f"Neurons file not found: {neurons_file}\n"
                f"Please place neurons.csv in data/raw/"
            )

        if verbose:
            print(f"  Loading {neurons_file.name}...")

        neurons = pd.read_csv(neurons_file)

        if 'root_id' in neurons.columns and 'Root ID' not in neurons.columns:
            neurons = neurons.rename(columns={'root_id': 'Root ID'})

        if verbose:
            print(f"    ✓ {len(neurons):,} neurons")
            if 'cell_class' in neurons.columns:
                print(f"    ✓ {neurons['cell_class'].nunique()} cell classes")

        return neurons

    def create_name_mapping(self, verbose: bool = True) -> pd.DataFrame:
        """
        Create mapping from Root ID to cell names.

        Returns
        -------
        pd.DataFrame
            Mapping with columns: root_id, cell_name, display_name
        """
        if self.neurons is None:
            raise ValueError("Must load neurons data first")

        if verbose:
            print("  Creating name mapping...")

        id_col = 'Root ID' if 'Root ID' in self.neurons.columns else 'root_id'

        if id_col not in self.neurons.columns:
            raise ValueError("Could not find root ID column in neurons data")

        mapping = pd.DataFrame()
        mapping['root_id'] = self.neurons[id_col]

        for col in ['Primary Cell Type', 'cell_type', 'hemibrain_type', 'type',
                    'Alternative Cell Type(s)', 'Sub Class', 'Class']:
            if col in self.neurons.columns:
                mapping['cell_name'] = self.neurons[col]
                break

        for class_col in ['Class', 'cell_class']:
            if class_col in self.neurons.columns:
                mapping['cell_class'] = self.neurons[class_col]
                break

        for nt_col in ['Verified NT type', 'Predicted NT type', 'top_nt']:
            if nt_col in self.neurons.columns:
                mapping['neurotransmitter'] = self.neurons[nt_col]
                break

        if 'cell_name' in mapping.columns:
            mapping['display_name'] = mapping['cell_name'].fillna(
                mapping['root_id'].astype(str)
            )
        else:
            mapping['display_name'] = mapping['root_id'].astype(str)

        if verbose:
            named_count = mapping['display_name'].ne(mapping['root_id'].astype(str)).sum()
            print(f"    ✓ {named_count:,} neurons with names")

        # Build fast dict lookup
        self._name_lookup = dict(zip(mapping['root_id'], mapping['display_name']))

        return mapping

    def save_name_mapping(self, output_dir: str = 'data/processed') -> Path:
        """Persist the name mapping to disk."""
        if self.name_mapping is None:
            raise ValueError("Must create name mapping first")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        output_file = out / 'neuron_name_mapping.csv'
        self.name_mapping.to_csv(output_file, index=False)
        print(f"    ✓ Saved mapping to {output_file}")
        return output_file

    def get_neuron_name(self, root_id: int) -> str:
        """Get display name for a neuron (O(1) dict lookup)."""
        return self._name_lookup.get(root_id, str(root_id))

    def get_neuron_id(self, name: str) -> Optional[int]:
        """Get root ID from neuron name (exact, partial, then normalised match)."""
        if self.name_mapping is None:
            raise ValueError("Must create name mapping first")

        if 'cell_name' not in self.name_mapping.columns:
            return None

        cell_names = self.name_mapping['cell_name']

        # 1. Exact match
        match = self.name_mapping[cell_names == name]
        if len(match) > 0:
            return match['root_id'].iloc[0]

        # 2. Case-insensitive partial match
        match = self.name_mapping[
            cell_names.str.contains(name, case=False, na=False, regex=False)
        ]
        if len(match) > 0:
            return match['root_id'].iloc[0]

        # 3. Normalised match: strip underscores, hyphens, spaces then compare
        def _norm(s: str) -> str:
            return s.lower().replace('_', '').replace('-', '').replace(' ', '')

        name_norm = _norm(name)
        normalised = cell_names.dropna().apply(_norm)
        match_idx = normalised[normalised == name_norm].index
        if len(match_idx) > 0:
            return self.name_mapping.loc[match_idx[0], 'root_id']

        return None

    def filter_connections(
        self,
        min_synapses: Optional[int] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Filter connectivity data.

        Parameters
        ----------
        min_synapses : int, optional
            Minimum synapse count (paper used 3)
        """
        if self.connections is None:
            raise ValueError("Must load connections first")

        conn = self.connections.copy()
        original_len = len(conn)

        if min_synapses is not None:
            conn = conn[conn['weight'] >= min_synapses]

        if verbose and min_synapses:
            print(f"  Filtered connections (min {min_synapses} synapses):")
            print(f"    Before: {original_len:,}")
            print(f"    After:  {len(conn):,}")

        return conn

    def get_neuron_connectivity(
        self,
        neuron_id: int,
        connections: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Get downstream connectivity for a neuron."""
        if connections is None:
            if self.connections is None:
                raise ValueError("Must load connections first")
            connections = self.connections

        return connections[connections['source'] == neuron_id].copy()

    def add_names_to_dataframe(self, df: pd.DataFrame, id_columns: List[str]) -> pd.DataFrame:
        """Add name columns to a dataframe using vectorised dict lookup."""
        if self.name_mapping is None:
            raise ValueError("Must create name mapping first")

        df = df.copy()
        for col in id_columns:
            if col in df.columns:
                df[f"{col}_name"] = df[col].map(self._name_lookup).fillna(
                    df[col].astype(str)
                )

        return df
