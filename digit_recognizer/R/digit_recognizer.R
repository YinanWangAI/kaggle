# Scripts for Kaggle competition Digit Recognizer
library(data.table)
library(dplyr)
library(randomForest)
library(mxnet)

setwd("~/kaggle/digit_recognizer/")

load_data <- function(train_path = "data/train.csv",
                      test_path = "data/test.csv",
                      val_path = NA,
                      val_prop = 0.1,
                      seed = 0) {
  #' load data from raw file
  #' 
  #' @param train_path String, the path of training data
  #' @param test_path String, the path of test data
  #' @param val_path String, optional, the path of validation data, if not 
  #' supplied, the training data will be divided into training and validation
  #' data
  #' @param val_prop Float, from 0 to 1,if validation data is not supplied,
  #'  the val_prop proportion of training data will serve as validation data
  #'  @param seed Int, arguments for set.seed()
  #'  @return list with three elements (training, test and validation data)
  train <- fread(train_path)
  test <- fread(test_path)
  if (is.na(val_path)) {
    if (val_prop < 0 | val_prop >= 1) stop("Invalid validation 
                                           proportion (val_prop)")
    if (val_prop == 0) {
      warning("No validation data")
      return(list(train, test, NA))
    }
    set.seed(seed)
    train_number <- nrow(train)
    train_row <- sample(train_number, floor(train_number * (1 - val_prop)),
                        replace = F)
    val_row <- setdiff(1:train_number, train_row)
    val <- train[val_row, ]
    train <- train[train_row, ]
  } else {
    val <- fread(val_path)
  }
  return(list(train, test, val))
}

make_submission_file <- function(x, path) {
  submit <- data.table(
    ImageID = 1:length(x),
    Label = x
  )
  write.csv(submit, file = path, row.names = F)
}

# load training, validation and test data---------------------------------------
all_data <- load_data()
train <- as.data.table(data.frame(all_data[1]))
test <- as.data.table(data.frame(all_data[2]))
valid <- as.data.table(data.frame(all_data[3]))
train_all <- rbind(train, valid)

# benchmark validation error by random forest-----------------------------------
train_x <- as.matrix(train[, 2:ncol(train), with = F]) / 255  # scale to [0, 1]
train_y <- as.factor(train[[1]])
valid_x <- as.matrix(valid[, 2:ncol(valid), with = F]) / 255
valid_y <- as.factor(valid[[1]])
train_all_x <- as.matrix(train_all[, 2:ncol(train_all), with = F]) / 255
train_all_y <- as.factor(train_all[[1]])
test_x <- as.matrix(test)

num_trees_list <- c(10, 15, 20, 25, 30, 35, 50, 100, 200, 300)
for (num_trees in num_trees_list) {
  rf <- randomForest(x = train_x, y = train_y, 
                     xtest = valid_x, ytest = valid_y,
                     ntree = num_trees)
  rf_accuracy <- 1 - rf$test$err.rate[num_trees, 1]
  cat("The number of trees is: ", num_trees, 
      "accuracy on validation set is:", rf_accuracy, '\n')
}
num_trees_optim <- 200 # 0.9685714 on validation set
rf_optim <- randomForest(x = train_all_x, y = train_all_y,
                         xtest = test_x,
                         ntree = num_trees_optim) 
rf_pred <- rf_optim$test$predicted
make_submission_file(rf_pred, "data/submission_rf.csv")
  
# mlp---------------------------------------------------------------------------
layer1_num_list <- c(100, 200, 500, 800, 1000, 2000, 5000, 8000)
layer2_num_list <- c(100, 200, 500, 800, 1000, 2000, 5000, 8000)
accuracy_list <- list()
count <- 1
for (layer1_num in layer1_num_list) {
  for (layer2_num in layer2_num_list) {
    mlp <- mx.mlp(train_x, as.numeric(as.character(train_y)), 
                  hidden_node = c(layer1_num, layer2_num), out_node = 10,
                  learning.rate = 0.01, dropout = 0.5,
                  num.round = 20, momentum = 0.9)
    valid_y_prediction <- as.factor(apply(predict(mlp, valid_x), 2, which.max) - 1)
    mlp_accuracy <- sum(valid_y_prediction == valid_y) / length(valid_y)
    accuracy_list[count] <- c(layer1_num, layer2_num, mlp_accuracy)
    count <- count + 1
  }
}
mlp <- mx.mlp(train_x, as.numeric(as.character(train_y)), 
              hidden_node = c(3000, 2000), out_node = 10,
              learning.rate = 0.01, dropout = 0.5,
              num.round = 20, momentum = 0.9)
valid_y_prediction <- as.factor(apply(predict(mlp, valid_x), 2, which.max) - 1)
mlp_accuracy <- sum(valid_y_prediction == valid_y) / length(valid_y)
print(mlp_accuracy)  # 0.92

# convolutional network---------------------------------------------------------
train_array <- t(train_x)
dim(train_array) <- c(28, 28, 1, nrow(train_x))
valid_array <- t(valid_x)
dim(valid_array) <- c(28, 28, 1, nrow(valid_x))
train_all_array <- t(train_all_x)
dim(train_all_array) <- c(28, 28, 1, nrow(train_all_x))
test_array <- t(test_x)
dim(test_array) <- c(28, 28, 1, nrow(test_x))

data <- mx.symbol.Variable("data")
# first convolutional layer
conv1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 64)
conv1_activate <- mx.symbol.Activation(data = conv1, act_type = "relu")
pool1 <- mx.symbol.Pooling(data = conv1_activate, pool_type = "max",
                           kernel = c(2, 2), stride = c(2, 2))
# second convolutional layer
conv2 <- mx.symbol.Convolution(data = pool1, kernel = c(5, 5), num_filter = 32)
conv2_activate <- mx.symbol.Activation(data = conv2, act_type = "relu")
pool2 <- mx.symbol.Pooling(data = conv2_activate, pool_type = "max",
                           kernel = c(2, 2), stride = c(2, 2))
# first full connected layer
flatten <- mx.symbol.Flatten(pool2)
fc1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 20)
fc1_activate <- mx.symbol.Activation(data = fc1, act_type = "relu")
# output layer
fc2 <- mx.symbol.FullyConnected(data = fc1_activate, num_hidden = 10)
cost <- mx.symbol.SoftmaxOutput(data = fc2)
# train the model
cnn <- mx.model.FeedForward.create(
  cost, 
  X = train_array, y = as.numeric(as.character(train_y)),
  num.round = 10, array.batch.size = 50,
  learning.rate = 0.05, momentum = 0.9,
  eval.metric = mx.metric.accuracy,
  epoch.end.callback = mx.callback.log.train.metric(100))
cnn_pred <- predict(cnn, valid_array)
pred_label <- max.col(t(cnn_pred)) - 1
valid_accuracy <- sum(as.numeric(as.character(valid_y)) == pred_label) / 
  length(pred_label)
print(valid_accuracy)
# train model with all data
cnn <- mx.model.FeedForward.create(
  cost, 
  X = train_all_array, y = as.numeric(as.character(train_all_y)),
  num.round = 10, array.batch.size = 50,
  learning.rate = 0.05, momentum = 0.9,
  eval.metric = mx.metric.accuracy,
  epoch.end.callback = mx.callback.log.train.metric(100))
test_pred <- predict(cnn, test_array)
test_pred <- max.col(t(test_pred)) - 1
make_submission_file(test_pred, path = "data/submission_cnn.csv")
