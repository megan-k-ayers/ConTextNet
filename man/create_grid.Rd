% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/prep_data.R
\name{create_grid}
\alias{create_grid}
\title{Build parameter grid for tuning}
\usage{
create_grid(model_params, K, task)
}
\arguments{
\item{model_params}{List defining the parameter values to consider during
tuning.}

\item{K}{Number of folds (iterations) to train for each setting.}

\item{task}{Whether this is "reg" (regression) or "class" (classification).}
}
\value{
A tibble, where each row represents a different setting for model
parameters.
}
\description{
Build parameter grid for tuning
}
