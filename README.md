
<!-- README.md is generated from README.Rmd. Please edit that file -->

# ConTextNet

<!-- badges: start -->
<!-- badges: end -->

The goal of ConTextNet is to discover influential text features from a
corpus given an outcome of interest, according to the methodology
presented in Ayers et
al. ([2024](https://aclanthology.org/2024.findings-acl.714/)). These
text features can be used to estimate causal effects on the outcome
using a generalization of the framework presented in Fong & Grimmer
(2016, 2021), or can be viewed as exploratory evidence to guide
confirmatory analyses where researchers design a small number of
treatment texts. To do this, ConTextNet walks users through building,
training, tuning, and interpreting a neural network with convolutional
layers. ConTextNet relies heavily on R Keras for modeling, and will
likely require users to run at least some operations on a high
performance computing cluster (HPC) for corpora larger than a few
thousand documents.

In future development the package will provide documentation and helper
functions to guide users through the necessary Python dependency
installations and to streamline interactions with common HPC interfaces.

## Installation
You can install the development version of ConTextNet like so:
``` r
install_github("megan-k-ayers/ConTextNet")
```

## Example

This is an example of how to use the core functionality of the package.

``` r
library(ConTextNet)
set.seed(123)
tensorflow::set_random_seed(123)

### Setup
n <- 100
x <- read.csv("data-raw/imdb_full.csv")
x$y <- ifelse(x$y, 1, 0)
x <- x[sample(1:nrow(x), n), ]

# Create fake covariate(s) to test
x$cov1 <- rnorm(n)
# x$cov2 <- rnorm(n, mean = x$y, sd = 0.4)

model_params <- list("n_filts" = list(4), "kern_sizes" = list(5),
                     "lr" = list(0.001, 0.0001), "lambda_cnn" = list(0.001),
                     "lambda_corr" = list(0), "lambda_out" = list(0.005),
                     "epochs" = list(100), "batch_size" = list(32),
                     "patience" = 30, "covars" = list("cov1"))


### Prep and embed
inputs <- prep_data(x = x, y_name = "y", text_name = "text",
                    scale_cov = "min-max", scale_y = "normalize",
                    model_params = model_params, task = "class",
                    folder_name = NULL, tune_method = "local",
                    embed_instr = list(max_length = 40), folds = 3)
input_embeds <- embed(inputs)


### Run parameter tuning, take best ones w.r.t. validation MSE
dat <- input_embeds$dat; embeds <- input_embeds$embeds
meta_params <- input_embeds$params; grid <- input_embeds$grid;
tokens <- input_embeds$tokens; vocab <- input_embeds$vocab

tune_res <- tune_model(dat, embeds, meta_params, grid, tokens, vocab)

# Aggregating over runs of the same setting.
tune_res$act_range_avg <- apply(do.call(rbind, strsplit(tune_res$act_range, "|",
                                                        fixed = TRUE)),
                                1, function(a) mean(as.numeric(a)))
quant_cols <- grep("train|val|max_corr|avg", names(tune_res), value = TRUE)
tune_res_agg <- tune_res %>%
  group_by(id) %>%
  summarise(across(all_of(quant_cols), ~ mean(.x, na.rm = TRUE)),
            id = unique(id))

(tune_res_agg <- tune_res_agg[order(tune_res_agg$val_mse), ])
best_id <- tune_res_agg$id[1]
best_params <- c(meta_params,
                 get_row_list(tune_res[tune_res$id == best_id, ][1, ]))
input_embeds$params <- best_params


### Train final model with the "best" parameters
model <- train_model(dat[dat$fold == "train", ],
                     embeds[dat$fold == "train", , ], best_params)


### Assess the model quantitatively on the test set.
test_embeds <- embeds[dat$fold == "test", , ]
test_y <- dat$y[dat$fold == "test"]
test_cov <- as.matrix(dat[dat$fold == "test", best_params$covars])
eval_model(model, list(test_embeds, test_cov), test_y,
           metrics = c("mse", "f1", "accuracy"))
# eval_model(model, list(test_embeds), test_y,
#            metrics = c("mse"))


### Basic interpretation
p_acts <- get_phrase_acts(model, test_embeds, best_params,
                          dat[dat$fold == "test", ])
get_top_phrases(p_acts, tokens[dat$fold == "test", ], best_params, vocab,
                m = 5)

d_acts <- get_doc_acts(model, test_embeds, best_params,
                       dat[dat$fold == "test", ])
plot_doc_acts(d_acts)
```

<!-- What is special about using `README.Rmd` instead of just `README.md`? You can include R chunks like so: -->
<!-- ```{r cars} -->
<!-- summary(cars) -->
<!-- ``` -->
<!-- You'll still need to render `README.Rmd` regularly, to keep `README.md` up-to-date. `devtools::build_readme()` is handy for this. -->
<!-- You can also embed plots, for example: -->
<!-- ```{r pressure, echo = FALSE} -->
<!-- plot(pressure) -->
<!-- ``` -->
<!-- In that case, don't forget to commit and push the resulting figure files, so they display on GitHub and CRAN. -->
