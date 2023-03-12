pub mod data;
pub mod knn;
pub mod log;
pub mod perceptron;

use data::*;
use knn::*;

fn main() {
    let train_data = Knn::from_csv("misc/datasets/mnist_train.csv");
    let test_data = Knn::from_csv("misc/datasets/mnist_test.csv");
    let mut model = Knn::new().enable_log().set_topk(25);
    let acc = model.model_test(train_data, test_data);
    sparkle!("acc: {}", acc);
}
