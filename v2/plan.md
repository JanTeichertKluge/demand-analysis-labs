We need a new version of the labs for the Demand Analysis paper. Instead of using the full sample from the paper, we’d like to use only a subsample. To do this, the embeddings should be generated and fine-tuned live in a notebook (that runs on Google Colab!) using the DoubleMLDeep package (https://github.com/DoubleML/doubleml-deep). The full data package is in our reproducability github repo (see https://github.com/JanTeichertKluge/demand-analysis-repro/tree/main/main/data). The sub-sample size should be large enough but not too large, in order it must be executeable on google colab (with GPU support).

Notebooks we need:

1) Explore Data (keep v1 nb with new subsample)
2) Fine-Tuning of Embeddings (new)
3) Prediction (keep v1 nb with new subsample)
4) Elasticity (keep v1 with new subsample)
5) Heterogeneous Effects (new)
