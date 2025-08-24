flowchart TD
  %% =========================
  %% DATA & PREPROCESSING
  %% =========================
  subgraph A[Data & Preprocessing]
    RAW[Raw images]
    TAGJSON[Annotation JSON]
    ORIMAP[Orientation map JSON]
    H5[(HDF5 shards)]
    LMDB[(LMDB L2 cache)]
    HL[HDF5_loader.py<br/>• Dataloaders & augmentations<br/>• Letterbox resize to 640<br/>• Pad color 114<br/>• Normalize mean/std (0.5)]
    ORH[orientation_handler.py<br/>• Validate flip map<br/>• Remap left/right tags<br/>• Strict mode & fallback]
    RAW --> HL
    TAGJSON --> HL
    ORIMAP --> ORH --> HL
    H5 --> HL
    LMDB <--> HL
    HL --> BATCH[Batched tensors]
  end

  %% =========================
  %% VOCABULARY
  %% =========================
  subgraph B[Vocabulary]
    VOC[vocabulary.py<br/>TagVocabulary]
    VCHECK[verify_vocabulary_integrity<br/>• Placeholder detection<br/>• Hashing]
    VOC --> VCHECK
  end

  %% =========================
  %% MODEL & TRAINING
  %% =========================
  subgraph C[Model & Training]
    CFG[Configuration_System<br/>FullConfig & ModelConfig<br/>image_size=640, patch_size=16]
    CREATE[model_architecture.create_model<br/>(ViT-style tagger)]
    TRAIN[Training + validation_loop<br/>• AMP & grad accumulation<br/>• Schedulers & early stop]
    CKPT[checkpoint.pt]
    META[model_metadata.ModelMetadata<br/>• embed_vocabulary (b64+gzip+sha256)<br/>• embed_preprocessing_params<br/>(mean/std, image_size, patch_size)]
    CFG --> CREATE --> TRAIN
    BATCH --> TRAIN
    VCHECK --> TRAIN
    TRAIN -->|save| CKPT --> META
  end

  %% =========================
  %% INFERENCE
  %% =========================
  subgraph D[Inference]
    ICONF[InferenceConfig]
    PREP[ImagePreprocessor<br/>Resize 640 → Tensor → Normalize]
    WRAP[ModelWrapper (SimplifiedTagger)]
    IDS[InferenceDataset]
    PRED[predict (FP16 optional)]
    PROC[ResultProcessor → schemas.PredictionOutput(JSON)<br/>RunMetadata(top_k, threshold, vocab_sha256,<br/>mean/std, image_size, patch_size)]
    ICACHE[(Inference cache)]
    CKPT -->|load & extract embedded vocab + preproc| WRAP
    ICONF --> PREP
    PREP --> IDS --> PRED --> PROC
    WRAP --> PRED
    PROC <--> ICACHE
  end

  %% =========================
  %% EXPORT
  %% =========================
  subgraph E[Export]
    ONNXX[ONNX_Export.ONNXExporter<br/>add metadata (vocab, mean/std,<br/>image_size, patch_size, sha256)]
    ONNXM[(model.onnx with metadata)]
    CKPT --> ONNXX
    VOC --> ONNXX
    ONNXX --> ONNXM
    ONNXM --> ORT[onnx_infer.py<br/>reads embedded metadata or falls back<br/>to external vocabulary → JSON output]
  end

  %% =========================
  %% MONITORING & LOGGING
  %% =========================
  subgraph F[Monitoring & Logging]
    MON[Monitor_log.py<br/>SystemMonitor & TrainingMonitor]
    QLOG[Queue logging → rotating/compressed files]
    TRAIN --> MON
    PRED --> MON
    MON --> QLOG
  end

  %% Styling
  style LMDB fill:#f5f5f5,stroke:#999,stroke-width:1px
  style CKPT fill:#fff,stroke:#333
