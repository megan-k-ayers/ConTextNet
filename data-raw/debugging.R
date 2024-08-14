### MRE for crashing issue when training model directly after embedding step
### (without a restart)

rm(list = ls())
library(devtools)
load_all()
set.seed(123)

### Pull tiny data set.
x <- read.csv("data-raw/imdb_full.csv")
x <- x[sample(1:nrow(x), 7500), ]
# cat(paste(x$text[x$y == 0], collapse = "\n\n"))
# cat(paste(x$text[x$y == 1], collapse = "\n\n"))
x$fold <- "train"

### Tokenize.
xfmr <- reticulate::import("transformers")
tknzr <- xfmr$AutoTokenizer$from_pretrained("prajjwal1/bert-tiny",
                                            use_fast = FALSE, from_tf = TRUE,
                                            clean_up_tokenization_spaces = TRUE)
token_res <- tknzr(x$text, padding = "max_length", truncation = TRUE,
                max_length = as.integer(100), return_tensors = "pt")
# tokens <- token_res$input_ids$numpy()  # Simulating how package handles tensors.
# mask <- token_res$attention_mask$numpy()
vocab <- names(tknzr$get_vocab())

### Embed.
# torch <- reticulate::import("torch")
# tokens <- torch$tensor(tokens, dtype = torch$int64)
# mask <- torch$tensor(mask, dtype = torch$int64)

embed_model <- xfmr$AutoModel$from_pretrained("prajjwal1/bert-tiny")
embeds <- embed_model(attention_mask = token_res$attention_mask,
                      input_ids = token_res$input_ids)
embeds <- embeds$last_hidden_state
embeds <- embeds$detach()$numpy()

### Run model.
model_params <- list(params = list("n_filts" = 8, "kern_sizes" = 5,
                                   "lr" = 0.001, "lambda_cnn" = 0.001,
                                   "lambda_corr" = 0, "lambda_out" = 0.005,
                                   "epochs" = list(100), "batch_size" = 32,
                                   "patience" = 30, "covars" = NULL,
                                   "embed_dim" = 128, "task" = "class",
                                   "n_tokens" = 100),
                     "embeds" = embeds, "dat" = x)
model <- train_model(model_params)

### Evaluate.
eval_model(model, embeds, x$y, c("accuracy", "mse", "f1"))

p_acts <- get_phrase_acts(model, embeds, model_params$params)
View(get_top_phrases(p_acts, token_res$input_ids$numpy(), model_params$params, vocab))

d_acts <- get_doc_acts(model, embeds, model_params$params, x)
plot_doc_acts(d_acts)
