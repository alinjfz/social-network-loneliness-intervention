# Dataset

This project uses the **Facebook Social Circles** dataset from Stanford SNAP.

## Download

The dataset is not included in this repository due to size. Download it directly from SNAP:

```bash
curl -o facebook_combined.txt.gz https://snap.stanford.edu/data/facebook_combined.txt.gz
gunzip facebook_combined.txt.gz
```

Then move `facebook_combined.txt` to the **project root** (same directory as `network_intervention.py`).

## Dataset Details

| Property | Value |
|---|---|
| Source | [Stanford SNAP](https://snap.stanford.edu/data/ego-Facebook.html) |
| Nodes | 4,039 |
| Edges | 88,234 |
| Type | Undirected, unweighted |
| Global clustering | ~0.605 |
| Avg shortest path | ~3.69 |

## Citation

J. Leskovec and J. McAuley. *Learning to Discover Social Circles in Ego Networks.*
Advances in Neural Information Processing Systems (NeurIPS), 2012.
