% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tune_model.R
\name{get_tune_metrics}
\alias{get_tune_metrics}
\title{Get Performance Metrics During Parameter Tuning}
\usage{
get_tune_metrics(model, embeds, dat, metrics, fold, covars = NULL)
}
\arguments{
\item{model}{Trained Keras model}

\item{embeds}{Word embeddings}

\item{dat}{Input text data and covariates and outcome}

\item{metrics}{Metrics to get from eval_model}

\item{fold}{Fold to assess}

\item{covars}{Covariates for this model}
}
\value{
Performance metrics in a data frame row.
}
\description{
Get Performance Metrics During Parameter Tuning
}
