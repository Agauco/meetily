#[derive(thiserror::Error, Debug)]
pub enum DiarizationError {
    #[error("speaker embedding model error: {0}")]
    Model(String),
    #[error("audio segment too short for a voice embedding ({got} samples, need at least {min})")]
    TooShort { got: usize, min: usize },
    #[error("model output '{0}' not found")]
    OutputMissing(String),
}

impl From<ort::Error> for DiarizationError {
    fn from(e: ort::Error) -> Self {
        DiarizationError::Model(e.to_string())
    }
}
