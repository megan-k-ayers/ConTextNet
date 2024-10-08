###############################################################################
###     CHECK BASIC DATA ASSUMPTIONS, FLAG ISSUES BEFORE STARTING MODELING.
###
### Runs:         Locally.
### Status:       Skeleton only.
### Priority:     Low.
### User facing:  No.
###############################################################################


#' Check outcome column
#'
#' @param x The input data frame
#' @param y_name Name of the outcome column
#'
#' @return A logical flag indicating if format check passed (TRUE)
#'
#' @examples
check_y <- function(y_name, x) {
  if (!y_name %in% names(x)) {
    stop("The given `y_name` is not a column of `x`.")
  }
}

#' Check text column
#'
#' @param text_name Name of the text column
#' @param x The input data frame
#'
#' @return A logical flag indicating if format check passed (TRUE)
#'
#' @examples
check_text <- function(text_name, x) {
  if (!text_name %in% names(x)) {
    stop("The given `text_name` is not a column of `x`.")
  }
}

#' Check that covariates exist.
#'
#' @param x Data frame
#' @param covs Covariate names
#'
#' @return No return value, throws an error if check fails.
#'
#' @examples
check_covs <- function(x, covs) {
  if (!covs %in% names(x)) {
    stop("At least one of the specified covariates cannot be found in `x`.")
  }
}

