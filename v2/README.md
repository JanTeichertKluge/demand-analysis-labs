# Demand Analysis Labs, v2

Five notebooks working through [*Adventures in Demand Analysis Using AI*](https://arxiv.org/abs/2501.00382)
(Bach, Chernozhukov, Klaassen, Spindler, Teichert-Kluge and Vijaykumar) end to end, on a
subsample small enough to run on a free Colab GPU.

**What changed from v1.** The v1 labs loaded product embeddings that had already been
fine-tuned elsewhere, so the transformer step was a black box. Here Lab 2 fine-tunes the
embeddings live, using [`DoubleMLDeep`](https://github.com/DoubleML/doubleml-deep), so the
whole chain from raw text and images to a heterogeneous elasticity is visible and
executable.

---

## The labs

| | Notebook | GPU | Runs on |
|---|---|---|---|
| 1 | [Explore Data](01_explore_data.ipynb) | no | raw panel and images |
| 2 | [Fine-Tuning the Embeddings](02_finetune_embeddings.ipynb) | **yes** | raw panel and images |
| 3 | [Predicting Prices and Quantities](03_predicting_price_and_quantities.ipynb) | no | Lab 2's artifact |
| 4 | [The Average Price Elasticity](04_causal_elasticities.ipynb) | no | Lab 2's artifact |
| 5 | [Heterogeneous Price Elasticities](05_heterogeneous_effects.ipynb) | no | Lab 2's artifact |

To open one in Colab, replace `github.com` with `colab.research.google.com/github` in the
notebook URL, for example
`https://colab.research.google.com/github/JanTeichertKluge/demand-analysis-labs/blob/main/v2/02_finetune_embeddings.ipynb`.

### The argument in one pass

1. Lab 1 regresses quantity on price and gets $\hat\delta \approx -0.2$, implying demand
   so inelastic it cannot be true. Quality and visibility are missing and are not columns
   in the table.
2. Lab 2 constructs them, turning descriptions and images into embeddings fine-tuned to
   predict price and quantity.
3. Lab 3 finds that the embeddings predict levels well and changes barely at all.
4. Lab 4 uses that result. The lagged state is the real confounder, and the elasticity
   lands near $-0.7$ in rank terms, about $-1.4$ in demand terms, stable across
   specifications.
5. Lab 5 shows that the average conceals a lot: price sensitivity varies systematically
   with what a product is and how popular it is.

---

## Running them

Run Lab 2 before Labs 3 to 5. Lab 1 is independent.

Colab gives every notebook its own machine, so `/content` does not carry across notebooks.
Lab 2 writes its artifact to Google Drive
(`MyDrive/demand_labs_v2/subsample_v2.parquet`, a few MB) and Labs 3 to 5 read it back
from there, so accept the Drive mount prompt in each notebook. If you would rather not use
Drive, Lab 2 also writes a local copy that you can download and re-upload.

### Sample size

Each product image lives in one of sixteen parquet part-files of roughly 90 MB, and a
part-file cannot be queried for a single product's image. The subsample is therefore drawn
from whatever image files you download. The three listed in Labs 1 and 2 cover 2,000
products, of which 911 have a complete 14-period panel. `N_PRODUCTS = 800` of those are
used, split into `N_FINETUNE = 300` for fine-tuning and 500 for estimation, which leaves
about 6,500 observations once lags are taken.

To work with more products, add further part-files from the same folder to `image_files`
and raise `N_PRODUCTS`. Change both Labs 1 and 2 together, since they rebuild the same
subsample independently and the draw is only reproducible if the constants match.

| image files | download | products with a full panel |
|---|---|---|
| 3 (default) | ~277 MB | 911 |
| 5 | ~455 MB | 1,540 |
| 8 | ~725 MB | 2,692 |
| 16 | ~1.2 GB | 4,529 |

The paper uses 7,226 products and 38,041 observations, so this is a genuine subsample.

The default takes roughly 25 minutes of T4 time in Lab 2, less on an L4 or A100. Labs 1
and 3 to 5 take a few minutes each on CPU. These figures are estimates, since the
fine-tuning run has not been benchmarked across GPU types.

### What the sample size costs you

Standard errors scale roughly with $1/\sqrt{n}$, so at 800 products they come out about
twice the paper's.

- Lab 4 is robust. The average elasticity is large relative to its standard error and
  reproduces the paper's value closely.
- Lab 5 is not. Individual cluster-similarity coefficients will often be insignificant
  where the paper finds significance. Read the joint $\chi^2$ test and the spread of
  $\hat\alpha(x)$ rather than individual $t$-statistics, or scale the sample up.

---

## Data

Everything is pulled at runtime from the paper's reproduction repository,
[`JanTeichertKluge/demand-analysis-repro`](https://github.com/JanTeichertKluge/demand-analysis-repro):

- `main/data/amzn_toys_monthly_diffs_ffill_fixed_splits_buybox/` holds the panel (~42 MB).
  `SALES_RANK` is already $\log(1/\text{rank})$ and `BUYBOX_PRICE` already
  $\log(\text{price})$; the labs rename them to `Q_t` and `P_bb_t`.
- `main/data/amzn_toys_monthly_avg_long_ffill_28_7_images/` holds the product images.

The labs keep `window == 28` and every fourth weekly date, giving 14 four-week periods
from 2023-01-30 to 2024-01-29, and keep only products observed in all of them. The loading
path is plain top-to-bottom code in Labs 1 and 2.

Products are split by ASIN into a fine-tune sample (the paper's $I_1$) and an estimation
sample ($I_2$). Labs 4 and 5 estimate only on $I_2$, so the embeddings entering the causal
analysis were fitted on different products, which is the independence condition behind the
paper's Addendum A. The split is deterministic (sorted candidates, fixed seed), which is
why Labs 1 and 2 rebuild identical subsamples without sharing a file.

---

## Notes on the environment

`doubleml-deep` imports its SAINT tabular model at package import time, so
`pytorch-widedeep` has to be importable even though the labs barely use it. Installing it
normally pins `numpy<2` and pulls in `spacy`, `gensim` and `opencv`, which breaks Colab's
pre-installed stack. Lab 2 therefore installs it with `--no-deps` and supplies two small
stub modules for the text utilities it imports but never calls. This is deliberate and
commented in the notebook.

To run locally instead of on Colab:

```bash
pip install -e ".[v2]"
pip install --no-deps pytorch-widedeep
pip install git+https://github.com/DoubleML/doubleml-deep.git
```

You will still need the `gensim` and `spacy` stubs from Lab 2's shim cell, or the real
packages.

---

## Verification status

Every notebook has been executed end to end, but not all of them against real inputs.

- Lab 1 runs on the real data as shipped.
- Lab 2 has been run with small stand-in encoders on CPU, which exercises the whole
  `doubleml-deep` path. It has not been run with the real RoBERTa, BEiT and SAINT encoders
  on a GPU, so its timings are estimates.
- Labs 3 to 5 were run against a stand-in artifact carrying the real panel but placeholder
  embedding columns. Their code paths are verified; any result attributable to the
  embeddings is not.

Because the elasticity is driven by the lagged state rather than the embeddings, Lab 4's
numbers under that stand-in are still meaningful, and they land close to the paper:
$-0.685$ and $-0.711$ for the two boosted-tree specifications, against $-0.697$ and
$-0.691$ in Table 7.

What remains is one GPU run of Lab 2 with the real encoders.
