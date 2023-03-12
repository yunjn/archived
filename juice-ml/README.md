<div align="center">

## juice

</div>

### Quick Start

* Perceptron

```Rust
let train_data = Perceptron::from_csv("assets/mnist_train.csv");
let test_data = Perceptron::from_csv("assets/mnist_test.csv");

let model = Perceptron::new()
    .set_epochs(50)
    .set_step_size(0.0001)
    .enable_log()
    .train(train_data);

let acc = model.test(test_data);
println!("acc: {}", acc);
```

* KNN

```Rust
let train_data = Knn::from_csv("assets/mnist_train.csv");
let test_data = Knn::from_csv("assets/mnist_test.csv");
let mut model = Knn::new().enable_log().set_topk(25);
let acc = model.model_test(train_data, test_data);
println!("acc: {}", acc);
```

* Test

```shell
cargo test  --release --  --show-output
```

### Implemented

- [x] Single Layer Perceptron
- [x] Perceptron
- [x] KNN
