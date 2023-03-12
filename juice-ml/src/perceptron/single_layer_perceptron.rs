type AF = Box<dyn Fn(f64) -> f64>;

#[derive(Debug, Default)]
pub struct DataSet {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
}

pub struct SingleLayerPerceptron {
    lr: f64,                 // 学习率
    baise: f64,              // 偏置
    epochs: usize,           // 代
    dim: usize,              // 维度
    w: Vec<f64>,             // 权值
    dataset: DataSet,        // 数据集
    activation_function: AF, // 激活函数
    logger: bool,            // 记录
}

impl SingleLayerPerceptron {
    pub fn new() -> Self {
        Self {
            epochs: 0,
            lr: 0.01,
            baise: 0.4,
            dim: 0,
            w: vec![0.0],
            logger: false,

            dataset: DataSet {
                x: vec![vec![0.0]],
                y: vec![0.0],
            },

            activation_function: Box::new(|x: f64| -> f64 {
                match x.partial_cmp(&0.0) {
                    Some(std::cmp::Ordering::Less) => -1.0,
                    _ => 1.0,
                }
            }),
        }
    }

    pub fn set_activation_function(self, activation_function: AF) -> Self {
        Self {
            activation_function,
            ..self
        }
    }

    pub fn set_epochs(self, epochs: usize) -> Self {
        Self { epochs, ..self }
    }

    pub fn set_learning_rate(self, lr: f64) -> Self {
        Self { lr, ..self }
    }

    pub fn set_baise(self, baise: f64) -> Self {
        Self { baise, ..self }
    }

    pub fn enable_log(self) -> Self {
        Self {
            logger: true,
            ..self
        }
    }

    pub fn load_dataset(self, dataset: DataSet) -> Self {
        if dataset.x.len() != dataset.y.len() {
            println!("x.len() != y.len()")
        }

        Self {
            dim: dataset.x[0].len(),
            w: vec![0.3; dataset.x[0].len()],
            dataset,
            ..self
        }
    }

    #[inline]
    fn comput_the_output(&self, x: &[f64]) -> f64 {
        let activation_function = self.activation_function.as_ref();

        activation_function({
            let mut tmp = 0.0;

            x.iter().zip(self.w.iter()).for_each(|(x_, w_)| {
                tmp += x_ * w_;
            });

            tmp + self.baise
        })
    }

    pub fn train(mut self) -> Self {
        for _ep in 0..self.epochs {
            let mut errors = 0.0;
            let mut _r = 0.0;
            let mut _output = 0.0;

            for (x, y) in self.dataset.x.iter().zip(self.dataset.y.iter()) {
                _output = self.comput_the_output(x);

                if _output == *y {
                    continue;
                }

                _r = self.lr * (y - _output);
                self.baise += _r;

                x.iter().zip(self.w.iter_mut()).for_each(|(x_, w_)| {
                    *w_ += _r * *x_;
                });

                errors += _r.abs();
            }

            if self.logger {
                println!(
                    "epoch: {} error: {:.5} baise: {:+.6} w: {:.8?}",
                    _ep, errors, self.baise, self.w
                );
            }

            if errors == 0.0 {
                break;
            }
        }
        Self { ..self }
    }

    pub fn predict(&self, x: &Vec<f64>) -> f64 {
        if x.len() != self.dim {
            println!("wrong dimension");
        }

        self.comput_the_output(x)
    }
}

#[test]
fn test_and() {
    let x = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let y = vec![0.0, 0.0, 0.0, 1.0];

    let and = Box::new(|x: f64| -> f64 {
        match x.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => 1.0,
            _ => 0.0,
        }
    });

    let model = SingleLayerPerceptron::new()
        .set_epochs(100)
        .set_learning_rate(0.01)
        .set_baise(0.4)
        .load_dataset(DataSet {
            x: x.clone(),
            y: y.clone(),
        })
        .set_activation_function(and)
        .train();

    let mut predicts: Vec<f64> = Vec::new();
    x.iter().for_each(|x_| predicts.push(model.predict(x_)));
    assert_eq!(predicts, y);
}

#[test]
fn test_or() {
    let x = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let y = vec![0.0, 1.0, 1.0, 1.0];

    let or = Box::new(|x: f64| -> f64 {
        match x.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => 1.0,
            _ => 0.0,
        }
    });

    let model = SingleLayerPerceptron::new()
        .set_epochs(100)
        .set_learning_rate(0.01)
        .set_baise(0.4)
        .load_dataset(DataSet {
            x: x.clone(),
            y: y.clone(),
        })
        .set_activation_function(or)
        .train();

    let mut predicts: Vec<f64> = Vec::new();
    x.iter().for_each(|x_| predicts.push(model.predict(x_)));
    assert_eq!(predicts, y);
}

#[test]
fn test_not() {
    let x = vec![vec![0.0], vec![1.0]];

    let y = vec![1.0, 0.0];

    let not = Box::new(|x: f64| -> f64 {
        match x.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => 1.0,
            _ => 0.0,
        }
    });

    let model = SingleLayerPerceptron::new()
        .set_epochs(100)
        .set_learning_rate(0.01)
        .set_baise(0.4)
        .load_dataset(DataSet {
            x: x.clone(),
            y: y.clone(),
        })
        .set_activation_function(not)
        .train();

    let mut predicts: Vec<f64> = Vec::new();
    x.iter().for_each(|x_| predicts.push(model.predict(x_)));
    assert_eq!(predicts, y);
}

#[test]
fn test_xor() {
    let x = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let y = vec![1.0, 0.0, 0.0, 1.0];

    let xor = Box::new(|x: f64| -> f64 {
        match x.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => 1.0,
            _ => 0.0,
        }
    });

    let model = SingleLayerPerceptron::new()
        .set_epochs(100)
        .set_learning_rate(0.01)
        .set_baise(0.4)
        .load_dataset(DataSet {
            x: x.clone(),
            y: y.clone(),
        })
        .set_activation_function(xor)
        .train();

    let mut predicts: Vec<f64> = Vec::new();
    x.iter().for_each(|x_| predicts.push(model.predict(x_)));
    assert_ne!(predicts, y);
}

#[test]
fn test_data() {
    let x = vec![
        vec![1.0, 0.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0],
    ];

    let y = vec![-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0];

    let model = SingleLayerPerceptron::new()
        .set_epochs(100)
        .set_learning_rate(0.01)
        .set_baise(0.3)
        .load_dataset(DataSet { x, y })
        // .enable_log()
        .train();

    let input = vec![0.0, 1.0, 1.0];
    let predict = model.predict(&input);
    assert_eq!(1.0, predict);
}
