# Forager Web

A web application for analyzing verbal fluency task (VFT) data. The website runs the [forager](https://github.com/thelexiconlab/forager) module with the selected switch method, returning results as a zip file containing CSVs.

## Dependencies

- Python 3.8 (dependencies in `Pipfile`)
  - Flask
  - Scipy
  - Pandas
  - NumPy
  - nltk
- AngularJS (frontend)

## How to run

```bash
python application.py
```

Open `http://localhost:8000` in your browser.

## Input File Format

Upload a delimited text file (CSV, TSV, TXT, etc.) with a header row. Columns are detected by name (case-insensitive), not by position.

| Column | Required | Aliases | Description |
|--------|----------|---------|-------------|
| `SID` | Yes | `id`, `subject`, `participant` | Participant ID |
| `entry` | Yes | `item`, `word`, `response` | Response item (word produced by the participant) |
| `timepoint` | No | | Timepoint identifier for grouping multiple lists per participant |
| `time` | No | `rt`, `response_time` | Response time for each item (required for Slope Difference and PEI methods) |

Example with required columns only:
```
SID,entry
1,dog
1,cat
1,horse
2,lion
2,tiger
```

Example with timing data:
```
SID,entry,time
1,dog,2.1
1,cat,4.3
1,horse,8.7
2,lion,1.9
2,tiger,5.2
```

Timing data can be provided as cumulative times or inter-response times (IRT), in seconds or milliseconds. These options are configurable in the web interface when a timing-based method is selected.

## Available Switch Methods

| Method | Parameters | Requires Timing |
|--------|-----------|-----------------|
| Similarity-drop (simdrop) | None | No |
| Multimodal Similarity-drop | alpha (0-1) | No |
| Norm-based associative | None | No |
| Norm-based categorical | None | No |
| Delta Similarity | rise threshold (0-1), fall threshold (0-1) | No |
| Slope Difference | None | Yes |
| Probabilistic Evidence Integration (PEI) | alpha (0-1), beta (0-1), prior (0.1-0.9; prior probability of clustering) | Yes |

For methods with parameters (multimodal, delta, PEI), you can either set specific parameter values or run a grid search over a range of values. Specific values can have any number of decimals but must be within the ranges above; if a value is missing or out of range, the page shows a message and does not submit, and the server rejects the request (HTTP 400) instead of using a default. When `all` methods are selected, the parametric methods always run over their full grids.

## Output

Results are returned as a zip file (`<input file name>_forager.zip`) containing:
- `switch_results.csv` — item-level switch designations (`Switch_Value`: 1 = switch, 0 = cluster, 2 = boundary case) for every selected method and parameter combination (`Switch_Method`). `PEI_Cluster_Probability` contains the PEI posterior probability of clustering, P(cluster), for PEI rows: an item is a switch when P(cluster) < 0.5, and P(switch) = 1 − P(cluster). It is empty (NaN) for the first item of each list, PEI boundary cases and rows of other methods
- `lexical_results.csv` — semantic similarity, phonological similarity, and frequency values. When timing data is provided, also includes cumulative response time (`Cumulative_IRT`), inter-response time (`IRT`), and slope difference (`Slope_Difference`)
- `individual_descriptive_stats.csv` — individual-level statistics, including the number of switches and the mean/SD of cluster size per switch method (a switch item starts a new cluster)
- `aggregate_descriptive_stats.csv` — group-level statistics
- `evaluation_results.csv` — details on OOV word handling (replacements, exclusions, truncations)
- `processed_data.csv` — the processed dataset used in the pipeline
- `forager_vocab.csv` — the full vocabulary used by forager

The CSV files do not include a row-index column, matching the output of the forager command-line package.

The Get Lexical Values option returns the same files without `switch_results.csv` and `aggregate_descriptive_stats.csv`.

## Lexical data

The files in `data/lexical_data/` (vocabulary, embeddings, semantic and phonological similarity matrices, frequencies) and `data/norms/animals_snafu_scheme_vocab.csv` are copies of the forager package's `animals` data (`data/lexical_data/animals/` and `data/norms/` in [forager](https://github.com/thelexiconlab/forager)), so the web interface and the command-line package give the same results. Word frequencies are Zipf frequencies from the `wordfreq` package (`forager/frequency.py`; `wordfreq` is only needed to regenerate them). The vocabulary includes an `UNK` entry used by the "Assign mean vector to OOV" option. When the forager data changes, copy the files again.

## Changes from forager package

- `run_foraging.py` is replaced by `run_foraging_web.py`
- `prepareData()` in `utils.py` is replaced by `evaluate_web_data()` for non-interactive data handling
- `cues.py` adds `get_oov_sims()` for on-the-fly OOV embedding generation
- Parametric switch methods support specific values or grid search

## Running Tests

```bash
python -m pytest forager/tests/test_switches.py
```
