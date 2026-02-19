use std::default::Default;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use crate::storage::{Storage, CpuStorage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    U8,
    U16,
    U32,
    I32,
    F32,
    F64,
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::U8 => 1,
            DataType::U16 => 2,
            DataType::U32 | DataType::I32 | DataType::F32 => 4,
            DataType::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorShape {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl TensorShape {
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
        }
    }

    pub fn hw(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    pub fn chw(&self) -> (usize, usize, usize) {
        (self.channels, self.height, self.width)
    }

    pub fn len(&self) -> usize {
        self.channels * self.height * self.width
    }

    pub fn is_1d(&self) -> bool {
        self.height == 1 && self.width == 1
    }

    pub fn is_2d(&self) -> bool {
        self.channels == 1
    }

    pub fn is_3d(&self) -> bool {
        self.channels > 1
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<T: Clone + Copy + 'static, S: Storage<T> = CpuStorage<T>> {
    pub storage: S,
    pub shape: TensorShape,
    pub dtype: DataType,
    pub _phantom: PhantomData<T>,
}

pub type CpuTensor<T> = Tensor<T, CpuStorage<T>>;

impl<T: Clone + Copy + fmt::Debug + 'static, S: Storage<T>> Tensor<T, S> {
    pub fn from_vec(data: Vec<T>, shape: TensorShape) -> Self {
        assert_eq!(data.len(), shape.len(), "Data size mismatch with shape");
        let dtype = match std::any::type_name::<T>() {
            "u8" => DataType::U8,
            "u16" => DataType::U16,
            "u32" => DataType::U32,
            "i32" => DataType::I32,
            "f32" => DataType::F32,
            "f64" => DataType::F64,
            _ => DataType::F32,
        };
        Self {
            storage: S::from_vec(data),
            shape,
            dtype,
            _phantom: PhantomData,
        }
    }

    pub fn reshape(&self, new_shape: TensorShape) -> Self {
        let len = self.storage.len();
        
        assert_eq!(
            len,
            new_shape.len(),
            "Cannot reshape: size mismatch"
        );
        Self {
            storage: self.storage.clone(),
            shape: new_shape,
            dtype: self.dtype,
            _phantom: PhantomData,
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice().expect("Data not on CPU")
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice().expect("Data not on CPU")
    }

    pub fn index(&self, c: usize, h: usize, w: usize) -> T {
        let idx = c * self.shape.height * self.shape.width + h * self.shape.width + w;
        self.as_slice()[idx]
    }

    pub fn index_mut(&mut self, c: usize, h: usize, w: usize) -> &mut T {
        let idx = c * self.shape.height * self.shape.width + h * self.shape.width + w;
        &mut self.as_mut_slice()[idx]
    }

    /// Create a new tensor with the same metadata but different storage.
    pub fn with_storage<S2: Storage<T>>(self, storage: S2) -> Tensor<T, S2> {
        Tensor {
            storage,
            shape: self.shape,
            dtype: self.dtype,
            _phantom: PhantomData,
        }
    }

    /// Attempt to map the storage of this tensor to another type using a closure.
    pub fn try_map_storage<S2: Storage<T>, E, F>(self, f: F) -> std::result::Result<Tensor<T, S2>, E>
    where
        F: FnOnce(S) -> std::result::Result<S2, E>,
    {
        Ok(Tensor {
            storage: f(self.storage)?,
            shape: self.shape,
            dtype: self.dtype,
            _phantom: PhantomData,
        })
    }
}

impl<T: Clone + Copy + Default + fmt::Debug + 'static> Tensor<T> {
    pub fn new(shape: TensorShape) -> Self {
        let dtype = match std::any::type_name::<T>() {
            "u8" => DataType::U8,
            "u16" => DataType::U16,
            "u32" => DataType::U32,
            "i32" => DataType::I32,
            "f32" => DataType::F32,
            "f64" => DataType::F64,
            _ => DataType::F32,
        };

        Self {
            storage: CpuStorage::new(shape.len(), T::default()),
            shape,
            dtype,
            _phantom: PhantomData,
        }
    }

    pub fn zeros(shape: TensorShape) -> Self {
        Self::new(shape)
    }

    pub fn ones(shape: TensorShape) -> Self {
        let mut t = Self::new(shape);
        if let Some(s) = t.storage.as_mut_slice() {
            s.fill(T::default());
        }
        t
    }
}

impl Tensor<f32> {
    pub fn from_image_gray(data: &[u8], width: usize, height: usize) -> Self {
        let mut float_data = Vec::with_capacity(width * height);
        for &pixel in data {
            float_data.push(pixel as f32 / 255.0);
        }
        Self::from_vec(float_data, TensorShape::new(1, height, width))
    }

    pub fn from_image_rgb(data: &[u8], width: usize, height: usize) -> Self {
        let mut float_data = Vec::with_capacity(3 * width * height);
        for chunk in data.chunks(3) {
            float_data.push(chunk[0] as f32 / 255.0);
            float_data.push(chunk[1] as f32 / 255.0);
            float_data.push(chunk[2] as f32 / 255.0);
        }
        Self::from_vec(float_data, TensorShape::new(3, height, width))
    }

    pub fn to_image_gray(&self) -> Vec<u8> {
        assert!(self.shape.is_2d(), "Tensor must be 2D for gray image");
        self.as_slice()
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect()
    }
}

impl<T: Clone + Copy + fmt::Debug + 'static, S: Storage<T>> fmt::Display for Tensor<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor({}, {}, {}, {:?})",
            self.shape.channels, self.shape.height, self.shape.width, self.storage.device()
        )
    }
}

// Deref coercion for CpuStorage tensors
impl<T: Clone + Copy + fmt::Debug + 'static> Deref for Tensor<T, CpuStorage<T>> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Clone + Copy + fmt::Debug + 'static> DerefMut for Tensor<T, CpuStorage<T>> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

pub type Tensor3f = Tensor<f32>;
pub type Tensor4f = Tensor<f64>;

pub fn create_tensor_2d<T: Clone + Copy + Default + fmt::Debug + 'static>(height: usize, width: usize) -> Tensor<T> {
    Tensor::new(TensorShape::new(1, height, width))
}

pub fn create_tensor_3d<T: Clone + Copy + Default + fmt::Debug + 'static>(
    channels: usize,
    height: usize,
    width: usize,
) -> Tensor<T> {
    Tensor::new(TensorShape::new(channels, height, width))
}
