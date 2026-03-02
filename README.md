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
| Probabilistic Evidence Integration (PEI) | alpha (0-1), beta (0-1), prior (0.1-0.9) | Yes |

For methods with parameters (multimodal, delta, PEI), you can either set specific parameter values or run a grid search over a range of values.

## Output

Results are returned as a zip file containing:
- `switch_results_<method>.csv` — one CSV per switch method family with switch designations
- `lexical_results.csv` — semantic similarity, phonological similarity, and frequency values
- `individual_descriptive_stats.csv` — individual-level statistics
- `aggregate_descriptive_stats.csv` — group-level statistics
- `evaluation_results.csv` — details on OOV word handling (replacements, exclusions, truncations)
- `processed_data.csv` — the processed dataset used in the pipeline
- `forager_vocab.csv` — the full vocabulary used by forager

## Changes from forager package

- `run_foraging.py` is replaced by `run_foraging_web.py`
- `prepareData()` in `utils.py` is replaced by `evaluate_web_data()` for non-interactive data handling
- `cues.py` adds `get_oov_sims()` for on-the-fly OOV embedding generation
- `frequency.py` uses Google Books Ngram API instead of the `wordfreq` package
- Parametric switch methods support specific values or grid search

## Running Tests

```bash
python -m pytest forager/tests/test_switches.py
```
