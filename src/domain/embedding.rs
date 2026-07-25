/// Number of inputs sent per `/embeddings` request when embedding a batch.
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// How a batch of inputs should be turned into vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingOptions {
    /// Inputs per HTTP request. Larger batches cut round-trips; too large and
    /// servers reject the payload.
    pub batch_size: usize,
    /// L2-normalise each vector so cosine similarity equals the dot product.
    /// On by default — vector stores overwhelmingly assume it.
    pub normalize: bool,
    /// Requested output dimensionality, for models that support truncation
    /// (OpenAI's `text-embedding-3-*`). `None` leaves the model's native size.
    pub dimensions: Option<usize>,
}

impl Default for EmbeddingOptions {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            normalize: true,
            dimensions: None,
        }
    }
}

impl EmbeddingOptions {
    /// Override the batch size. Zero is meaningless and would loop forever, so
    /// it falls back to the default.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = if batch_size == 0 {
            DEFAULT_BATCH_SIZE
        } else {
            batch_size
        };
        self
    }

    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
}

/// Scale `vector` to unit length in place. A zero vector has no direction, so
/// it is left untouched rather than producing `NaN`s.
pub fn l2_normalize(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_to_unit_length() {
        let mut vector = vec![3.0, 4.0];
        l2_normalize(&mut vector);
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn leaves_the_zero_vector_alone() {
        // Dividing by a zero norm would yield NaNs and poison every later
        // similarity computation.
        let mut vector = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut vector);
        assert_eq!(vector, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn zero_batch_size_falls_back_to_the_default() {
        // A zero batch size would make the chunking loop make no progress.
        assert_eq!(
            EmbeddingOptions::default().with_batch_size(0).batch_size,
            DEFAULT_BATCH_SIZE
        );
        assert_eq!(EmbeddingOptions::default().with_batch_size(8).batch_size, 8);
    }

    #[test]
    fn normalization_is_on_by_default() {
        assert!(EmbeddingOptions::default().normalize);
    }
}
