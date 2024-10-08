% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tune_model.R
\name{get_doc_metrics}
\alias{get_doc_metrics}
\title{Get Document-Level Activation Metrics (tuning)}
\usage{
get_doc_metrics(model, params, embeds, dat)
}
\arguments{
\item{model}{R Keras trained model}

\item{params}{Model parameter list}

\item{embeds}{Matrix of word embeddings to get activations for}

\item{dat}{Original data set}
}
\value{
Data frame with metrics.
}
\description{
Get Document-Level Activation Metrics (tuning)
}
