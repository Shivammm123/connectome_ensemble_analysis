# Flight Control Circuit Analysis Pipeline
## Systematic Reconstruction of Descending Neuron Pathways in *Drosophila* BANC Connectome

---

## Overview

This pipeline performs comprehensive circuit reconstruction and analysis of flight control pathways in the *Drosophila melanogaster* ventral nerve cord (VNC) using the FlyWire BANC connectome dataset. The analysis systematically maps complete pathways from brain descending neurons (DNs) through premotor interneuron (IN) modules to motor neurons (MNs) and muscles.

**Key Discoveries:**
- 5 functional premotor modules (1 power, 4 steering)
- 2,080 premotor interneurons organized hierarchically
- 829 hub neurons identified
- Complete DN→IN→MN→Muscle pathway database
- Network modularity score: 0.35 (moderate functional separation)

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Data Requirements](#data-requirements)
3. [Pipeline Overview](#pipeline-overview)
4. [Analysis Steps](#analysis-steps)
5. [Key Methodology Parameters](#key-methodology-parameters)
6. [Output Structure](#output-structure)
7. [Running the Analysis](#running-the-analysis)
8. [Results Interpretation](#results-interpretation)
9. [Citation & References](#citation--references)

---

## Installation & Setup

### Requirements

**Python 3.8+** with the following packages:

```bash
pip install pandas numpy scipy matplotlib seaborn networkx scikit-learn pyyaml plotly
```

### Directory Structure

```
connectome_ensemble_analysis/
├── config.yaml                 # Configuration file
├── data/
│   └── raw/
│       ├── connections_princeton.csv  # ~192 MB, 3.7M connections
│       └── neurons.csv                # ~115K neurons with annotations
├── scripts/
│   ├── utils/
│   │   ├── data_loader.py
│   │   └── similarity.py
│   ├── cell_type_classification.py
│   ├── muscle_mn_mapping.py
│   ├── interneuron_hub_clustering.py
│   ├── dn_pathway_mapping.py
│   ├── circuit_module_analysis.py
│   └── [additional analysis scripts]
└── results/
    └── [generated output directories]
```

---

## Data Requirements

### Source Dataset

**FlyWire BANC Connectome (Princeton)**
- **Total neurons:** 115,151
- **Connections:** 3.7 million synaptic connections
- **Resolution:** Synaptic-level connectivity
- **Coverage:** Complete adult VNC

### Required Files

1. **connections_princeton.csv**
   - Columns: `source`, `target`, `weight`
   - Source/target: Neuron IDs (Root ID)
   - Weight: Synapse count

2. **neurons.csv**
   - Columns: `Root ID`, `Super Class`, `Class`, `Function`, etc.
   - Contains cell type annotations and metadata

---

## Pipeline Overview

```
Step 1: Cell Type Classification
    ↓
Step 2: Motor Neuron Organization
    ↓
Step 3: Interneuron Hub Clustering ⭐ (Core Analysis)
    ↓
Step 4: DN Pathway Mapping
    ↓
Step 5: Circuit Module Analysis
    ↓
Step 6: Visualization & Summary
```

---

## Analysis Steps

### Step 1: Cell Type Classification
**Script:** `cell_type_classification.py`

**Purpose:** Classify neurons into functional categories

**Method:**
- Classification based on `Super Class` annotation
- Categories: DN, IN (VNC), MN, AN, Sensory, Glia
- Wing MN identification by function/class keywords

**Parameters:**
```yaml
# No parameters - uses annotations directly
```

**Output:**
- `all_neurons_classified.csv` (115,151 neurons)
- `descending_neurons.csv` (1,313 DNs)
- `vnc_interneurons.csv` (12,759 INs)
- `wing_motor_neurons.csv` (60 wing MNs)

---

### Step 2: Motor Pool Mapping
**Script:** `muscle_mn_mapping.py`

**Purpose:** Organize motor neurons by muscle targets

**Method:**
- Pattern matching on cell annotations
- Power muscles: DLM (dorsal longitudinal), DVM (dorsoventral)
- Steering muscles: basalar (b1-3), axillary (i1-2, III1-4), haltere (hg1-4)

**Parameters:**
```python
# Muscle classification patterns (regex)
power_patterns = {
    'DLM': r'DLM|dorsal.longitudinal',
    'DVM': r'DVM|dorsal.?ventral|dorsoventral'
}

steering_patterns = {
    'basalar': r'\bb[123]\b|basalar',
    'axillary_III': r'III[1-4]|third.axillary',
    'axillary_i': r'\bi[12]\b|first.axillary',
    'haltere': r'hg[1-4]|haltere',
    'pleurosternal': r'ps1|pleurosternal'
}
```

**Output:**
- `motor_pools.csv` (18 pools)
- `wing_mns_with_muscles.csv` (60 MNs with muscle assignments)

---

### Step 3: Interneuron Hub Clustering ⭐
**Script:** `interneuron_hub_clustering.py`

**Purpose:** Core analysis - discover functional premotor modules

**Method:**

1. **Find Premotor INs:**
   - Filter: IN → MN connections with ≥ min_synapses
   
2. **Build Connectivity Matrix:**
   - Rows: Interneurons (2,080)
   - Columns: Motor pools (18)
   - Values: Synapse counts

3. **Hierarchical Clustering:**
   - Similarity: Cosine similarity on connectivity vectors
   - Linkage: Ward's method
   - Distance: 1 - cosine_similarity

4. **Hub Identification:**
   - Degree centrality
   - Betweenness centrality
   - Number of MN targets

**Key Parameters:**
```yaml
min_synapses: 3              # Minimum synapses for valid connection
n_clusters: 5                # Number of functional modules
cluster_method: 'ward'       # Hierarchical clustering method
top_n_hubs: 20              # Top hub neurons to report
```

**Clustering Algorithm:**
```python
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity

# Build connectivity matrix
conn_matrix = build_connectivity_vectors(connections, in_ids, motor_pools)

# Compute similarity
similarity = cosine_similarity(conn_matrix)
distance = 1 - similarity

# Hierarchical clustering
linkage_matrix = linkage(distance, method='ward')
clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
```

**Output:**
- `premotor_interneurons_clustered.csv` (2,080 INs with cluster assignments)
- `cluster_characteristics.csv` (5 clusters)
- `hub_interneurons.csv` (2,080 INs ranked by centrality)
- `in_to_mn_connections.csv` (14,077 connections)

---

### Step 4: DN Pathway Mapping
**Script:** `dn_pathway_mapping.py`

**Purpose:** Map DN → IN → MN pathways

**Method:**

1. **Find DN→IN Connections:**
   - Filter: DN → premotor IN with ≥ min_synapses
   
2. **Classify DN Specialization:**
   - Primary cluster: Cluster receiving most DN input
   - Specialization: Inherited from primary cluster type

3. **Pathway Reconstruction:**
   - Trace: DN → IN → MN → Muscle
   - Calculate pathway strength: DN-IN synapses × IN-MN synapses

**Key Parameters:**
```yaml
min_synapses: 3              # Minimum DN→IN synapses
top_n_dns: 20               # Top DNs to analyze in detail
```

**DN Classification Logic:**
```python
# For each DN:
primary_cluster = cluster with max(DN→IN synapses)
specialization = primary_cluster.functional_type

# Categories:
# - power_control: DN primarily targets power module
# - steering_control: DN primarily targets steering modules
```

**Output:**
- `dn_classifications.csv` (DNs with specializations)
- `dn_to_in_connections.csv` (DN→IN connections)
- `complete_pathways.csv` (DN→IN→MN→Muscle chains)

---

### Step 5: Direct DN Pathways
**Script:** `dn_complete_pathway_analysis.py`

**Purpose:** Identify direct DN→MN connections (bypass INs)

**Method:**
- Find: DN → wing MN connections (≥ min_synapses)
- Compare: Direct vs indirect pathway usage
- Classify strategies: both, direct_only, indirect_only

**Key Parameters:**
```yaml
min_synapses: 3              # Minimum direct DN→MN synapses
```

**Strategy Classification:**
```python
for each DN:
    uses_direct = has DN→MN connections
    uses_indirect = has DN→IN→MN pathways
    
    if both:
        strategy = 'both'
    elif direct only:
        strategy = 'direct_only'
    elif indirect only:
        strategy = 'indirect_only'
```

**Output:**
- `dn_pathway_strategies.csv` (DN pathway classifications)
- `dn_to_mn_direct.csv` (direct connections)

---

### Step 6: Circuit Module Analysis
**Script:** `circuit_module_analysis.py`

**Purpose:** Quantify modularity and integration

**Method:**

1. **Modularity Score:**
   - Algorithm: Newman's modularity (NetworkX)
   - Range: -0.5 to 1.0 (higher = more modular)

2. **Integrative Neurons:**
   - Definition: INs connecting to ≥2 muscle types
   - Identifies: Power-steering coordination layer

3. **Statistical Validation:**
   - Power vs steering comparison
   - Within-cluster vs between-cluster connectivity

**Key Parameters:**
```yaml
min_clusters_for_integration: 2    # Min muscle types for "integrative"
min_similarity_integration: 0.3    # Min similarity threshold
```

**Modularity Calculation:**
```python
from networkx.algorithms import community

# Build IN network (edges = shared MN targets)
G = build_in_network(in_mn_connections)

# Compute modularity
communities = [set(cluster_INs) for cluster in clusters]
modularity = community.modularity(G, communities)
```

**Output:**
- `cluster_properties_detailed.csv` (per-cluster statistics)
- `integrative_interneurons.csv` (335 integration neurons)
- `modularity_score.txt` (0.3545)

---

## Key Methodology Parameters

### Global Parameters (config.yaml)

```yaml
# Connection filtering
min_synapses: 3                    # Minimum synapses for valid connection
                                   # Rationale: Reduces noise, focuses on strong connections
                                   # Literature: Standard threshold in connectomics

# Similarity analysis
normalization: 'none'              # No normalization (paper method)
                                   # Options: 'none', 'l1', 'l2', 'max'
similarity_metric: 'cosine'        # Cosine similarity
                                   # Formula: cos(θ) = A·B / (||A|| ||B||)

# Clustering
n_clusters: 5                      # Number of functional modules
                                   # Determined by: Dendrogram + biological validation
cluster_method: 'ward'             # Ward's linkage for hierarchical clustering
                                   # Rationale: Minimizes within-cluster variance

# Hub identification  
hub_centrality_threshold: 'median' # Neurons above median centrality = hubs
                                   # Alternative: Can set absolute threshold

# Pathway reconstruction
top_n_similar: 20                  # Number of similar neurons to consider
max_pathway_length: 3              # DN→IN→MN (3 steps)
```

### Critical Thresholds

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **min_synapses** | 3 | Noise reduction; standard in field |
| **n_clusters** | 5 | Optimal separation in dendrogram |
| **cluster_method** | ward | Minimizes variance; stable results |
| **cosine_similarity** | - | Direction matters more than magnitude |
| **integration_threshold** | 2 types | Connects ≥2 muscle types |

### Similarity Computation

**Connectivity Vector:**
```python
# For each neuron N:
connectivity_vector[N] = [synapses to target_1, 
                         synapses to target_2, 
                         ..., 
                         synapses to target_M]

# Example for IN connecting to 3 MNs:
IN_123: [0, 0, 5, 0, 12, 0, 8, ...]  # Length = total MNs
        ↑     ↑     ↑      ↑
       MN1   MN3   MN5    MN7
```

**Cosine Similarity:**
```python
similarity(A, B) = (A · B) / (||A|| × ||B||)

# Range: 0 to 1
# 1.0 = identical connectivity pattern
# 0.0 = no shared targets
# 0.5 = moderate similarity
```

---

## Output Structure

```
results/
├── cell_types/
│   ├── all_neurons_classified.csv
│   ├── descending_neurons.csv
│   ├── vnc_interneurons.csv
│   ├── wing_motor_neurons.csv
│   └── figures/
│       └── cell_type_summary.png
│
├── motor_neurons/
│   ├── motor_pools.csv
│   ├── wing_mns_with_muscles.csv
│   └── figures/
│       └── motor_neuron_organization.png
│
├── interneuron_clusters/
│   ├── premotor_interneurons_clustered.csv    # 2,080 INs
│   ├── cluster_characteristics.csv             # 5 clusters
│   ├── hub_interneurons.csv                    # Ranked by centrality
│   ├── in_to_mn_connections.csv                # 14,077 connections
│   └── figures/
│       └── in_clustering_analysis.png
│
├── dn_pathways/
│   ├── dn_classifications.csv
│   ├── dn_pathway_strategies.csv
│   ├── dn_to_in_connections.csv
│   ├── dn_to_mn_direct.csv
│   ├── complete_pathways.csv
│   └── figures/
│       ├── dn_pathway_analysis.png
│       └── dn_pathway_comparison.png
│
├── circuit_modules/
│   ├── cluster_properties_detailed.csv
│   ├── integrative_interneurons.csv            # 335 INs
│   ├── modularity_score.txt                    # 0.3545
│   └── figures/
│       └── circuit_module_analysis.png
│
├── pathway_database/
│   ├── complete_pathways_database.csv          # All pathways
│   ├── pathway_summary_*.csv
│   ├── connectivity_matrix_*.csv
│   └── database_metadata.json
│
├── candidate_prioritization/
│   ├── all_interneurons_ranked.csv             # All 2,080 ranked
│   ├── top_50_candidates.csv                   # Top priorities
│   ├── experimental_suggestions.csv
│   └── tier_*_candidates.csv
│
└── final_figures/
    ├── DN_to_Module_Connectivity.png
    ├── DN_to_Module_Connectivity.pdf
    ├── Figure_1_Methods_Pipeline.png
    └── Figure_2_Results_Summary.png
```

---

## Running the Analysis

### Complete Pipeline (Run in order)

```bash
# Step 1: Cell type classification
python scripts/cell_type_classification.py

# Step 2: Motor neuron organization  
python scripts/muscle_mn_mapping.py

# Step 3: Interneuron clustering (CORE)
python scripts/interneuron_hub_clustering.py

# Step 4: DN pathway mapping
python scripts/dn_pathway_mapping.py

# Step 5: Complete pathway analysis
python scripts/dn_complete_pathway_analysis.py

# Step 6: Module analysis
python scripts/circuit_module_analysis.py

# Generate final figures
python scripts/create_clean_dn_figure.py
python scripts/create_methods_figure.py
python scripts/create_results_figure.py

# Generate comprehensive report
python scripts/final_summary_report.py
```

### Quick Start (Essential Steps Only)

```bash
# Minimum viable analysis
python scripts/cell_type_classification.py
python scripts/muscle_mn_mapping.py
python scripts/interneuron_hub_clustering.py
python scripts/create_clean_dn_figure.py
```

---

## Results Interpretation

### Key Metrics

**Network Organization:**
- **Total premotor INs:** 2,080
- **Functional modules:** 5 (1 power, 4 steering)
- **Modularity score:** 0.35 (moderate separation)
- **Hub neurons:** 829 (40% of network)

**Module Characteristics:**

| Module | Type | INs | Synapses | Syn/IN | Primary Targets |
|--------|------|-----|----------|--------|-----------------|
| **Power** | Power | 370 | 49,333 | 133 | DLM1, DVM1A |
| **Steering A** | Steering | 335 | 17,446 | 52 | b1, b2, iii4 |
| **Steering B** | Steering | 539 | 32,973 | 61 | hg1, hg4, iii4 |
| **Steering C** | Steering | 460 | 23,911 | 52 | i1, i2, b3 |
| **Steering D** | Steering | 376 | 21,608 | 57 | ps1, iii1, iii3 |

**Power vs Steering:**
- **Steering has 1.94× MORE total synapses** (95,938 vs 49,333)
- **But power has 2.1× STRONGER connections per IN** (133 vs 56 syn/IN)
- **Interpretation:** Power = robust/reliable; Steering = distributed/flexible

**DN Pathway Strategies:**
- **Both pathways (DN→IN + DN→MN):** Most common
- **Indirect only (DN→IN→MN):** Majority of DNs
- **Direct only (DN→MN):** Rare, fast pathways

### Modularity Interpretation

**Score: 0.3545**

- **0.0-0.2:** Weak modularity (poorly separated)
- **0.2-0.4:** Moderate modularity ✅ **(Your result)**
- **0.4-0.6:** Strong modularity
- **>0.6:** Very strong modularity

**Biological meaning:**
- Power and steering are DISTINCT functional systems
- But maintain integration for coordinated flight
- Not completely separate (allows flexibility)

---

## Citation & References

### This Analysis

If you use this pipeline, please cite:

```
Flight Control Circuit Analysis Pipeline
FlyWire BANC Connectome Analysis
[Your Institution], 2025
```

### Data Source

```
FlyWire BANC Connectome (Princeton)
Takemura et al. (2024). A Connectome of the Male Drosophila Ventral Nerve Cord.
Nature. https://doi.org/10.1038/s41586-024-07389-x
```

### Methodology References

**Connectivity Analysis:**
```
Liessem et al. (2025). Descending control and regulation of 
spontaneous flight turns in Drosophila melanogaster.
Current Biology. https://doi.org/10.1016/j.cub.2024.11.002
```

**Premotor Circuits:**
```
Cheong et al. (2024). Transforming descending input into behavior: 
The organization of premotor circuits in the Drosophila Male Adult 
Nerve Cord connectome. eLife. https://doi.org/10.7554/eLife.96084
```

**Motor Organization:**
```
Azevedo et al. (2023). Synaptic architecture of leg and wing premotor 
control networks in Drosophila. Nature. 
https://doi.org/10.1038/s41586-024-07600-z
```

---

## Troubleshooting

### Common Issues

**Issue:** `FileNotFoundError: connections_princeton.csv`
- **Solution:** Ensure data files are in `data/raw/` directory

**Issue:** Module import errors
- **Solution:** Run scripts from project root directory

**Issue:** Out of memory during clustering
- **Solution:** Reduce `max_comparison_neurons` in config.yaml

**Issue:** No premotor INs found
- **Solution:** Check `min_synapses` threshold (try lowering to 2)

---

## Advanced Usage

### Modify Clustering Parameters

Edit `config.yaml`:
```yaml
# Try different cluster numbers
n_clusters: 6  # Instead of 5

# Try different linkage methods
cluster_method: 'complete'  # Options: ward, complete, average

# Adjust synapse threshold
min_synapses: 5  # Higher = stricter (fewer connections)
```

### Custom DN Analysis

```python
# In dn_pathway_mapping.py, modify:
top_n_dns = 50  # Analyze more DNs

# Or focus on specific DNs
specific_dns = ['DNa01', 'DNp03', 'DNb01']
```

### Extract Specific Pathways

```python
# Load pathway database
pathways = pd.read_csv('results/pathway_database/complete_pathways_database.csv')

# Filter for specific DN
dn_pathways = pathways[pathways['dn_name'].str.contains('DNa01')]

# Filter for power pathways
power_pathways = pathways[pathways['muscle_type'] == 'power']
```

---

## Performance Notes

**Typical Runtime (on standard laptop):**
- Step 1 (Classification): ~30 seconds
- Step 2 (Motor mapping): ~10 seconds
- Step 3 (IN clustering): ~2-3 minutes ⏰
- Step 4 (DN pathways): ~1-2 minutes
- Step 5 (Module analysis): ~1 minute
- **Total: ~5-10 minutes**

**Memory Requirements:**
- Minimum: 8 GB RAM
- Recommended: 16 GB RAM
- Peak usage: ~4-6 GB (during clustering)

---

## License

This analysis pipeline is provided for academic and research use.

---

## Contact & Support

For questions about:
- **Methodology:** See references above
- **Code issues:** Check troubleshooting section
- **Data access:** Contact FlyWire/Princeton team

---

## Acknowledgments

- **FlyWire Consortium** for BANC connectome data
- **Cheong, Azevedo, Liessem et al.** for methodological foundations
- **NetworkX, SciPy, scikit-learn** developers

---

**Last Updated:** March 2026
**Pipeline Version:** 1.0
**Python Version:** 3.8+

---
