% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/train_model.R
\name{eval_model}
\alias{eval_model}
\title{Evaluate a Trained ConTextNet Model's Performance}
\usage{
eval_model(model, input_dat, y, metrics)
}
\arguments{
\item{model}{A trained ConTextNet model, the output of train_model().}

\item{input_dat}{Input data to evaluate the model with. If covariates are
considered by the model, this should be a list with two elements:
the input embeddings and the input covariates.}

\item{y}{True labels corresponding to input_dat.}

\item{metrics}{Character array containing the metrics to evaluate the model
with (options include "mse", "accuracy", "f1").}
}
\value{
Data frame summarizing model performance.
}
\description{
Evaluate a Trained ConTextNet Model's Performance
}
\examples{
\dontrun{
model <- train_model(imdb_embed)
input_dat <- imdb_embed$embeds[imdb_embed$dat$fold == "test", , ]
y <- imdb_embed$dat$y[imdb_embed$dat$fold == "test"]
metrics <- c("mse", "accuracy", "f1")
eval_model(model, input_dat, y, metrics)
}
}
