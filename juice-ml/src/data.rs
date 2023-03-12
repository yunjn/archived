use ndarray::prelude::*;
use ndarray::{Array, Axis, Data, Dim, RemoveAxis};
use std::cmp::Ordering;
use std::path::Path;

pub struct LabeledDataset<D1, D2> {
    records: Array<f64, D1>,
    labels: Array<f64, D2>,
}

impl<D1: RemoveAxis, D2: RemoveAxis> LabeledDataset<D1, D2> {
    pub fn new(records: Array<f64, D1>, labels: Array<f64, D2>) -> Self {
        Self { records, labels }
    }

    /// Returns a reference to the records.
    pub fn records(&self) -> &Array<f64, D1> {
        &self.records
    }

    /// Returns a reference to the labels.
    pub fn labels(&self) -> &Array<f64, D2> {
        &self.labels
    }

    /// Returns the number of records stored in the labeled dataset.
    pub fn len(&self) -> usize {
        self.records.len_of(Axis(0))
    }

    /// Check whether the labeled dataset is empty or not.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub type L1D2 = LabeledDataset<Dim<[usize; 2]>, Dim<[usize; 1]>>;

pub trait LabeledDataLoader {
    fn from_csv(_path: impl AsRef<Path>) -> L1D2 {
        LabeledDataset::new(Array::default((0, 0)), Array::default(0))
    }
}

// 返回
pub fn argsort_by<S, F>(arr: &ArrayBase<S, Ix1>, mut compare: F) -> Vec<usize>
where
    S: Data,
    F: FnMut(&S::Elem, &S::Elem) -> Ordering,
{
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_unstable_by(move |&i, &j| compare(&arr[i], &arr[j]));
    indices
}
