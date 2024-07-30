#' Check outcome column
#'
#' @param x The input data frame
#'
#' @return A logical flag indicating if format check passed (TRUE)
#'
#' @examples
check_y <- function(x) {
  exists <- "y" %in% names(x)
  return(exists)
}

#' Check text column
#'
#' @return A logical flag indicating if format check passed (TRUE)
#'
#' @examples
check_text <- function() {
  return(1)
}

#' Check any co variate columns
#'
#' @return A logical flag indicating if format check passed (TRUE)
#'
#' @examples
check_normed <- function() {
  return(1)
}


#' Check input data format
#'
#' @return A logical flag indicating if all format checks passed (TRUE)
#' @export
#'
#' @examples
qa_data <- function() {
  return(1)
}
