flowchart TD
    subgraph Config["Configuration & Setup"]
        CFG["Configuration_System.py"]
        REQS["requirements.txt"]
        CONFIGS["configs/ (YAMLs/JSONs)"]
        IGNORES["Tags_ignore.txt"]
    end

    subgraph Data["Data Ingestion & Preparation"]
        LOADER["HDF5_loader.py\ letterbox_resize(),\ create_dataloaders()"]
        ORIENT["orientation_handler.py\ OrientationHandler"]
        METADATA["utils/metadata_ingestion.py\ parse_tags_field(), dedupe_preserve_order()"]
        VOCAB["vocabulary.py\ TagVocabulary:contentReference[oaicite:11]{index=11}"]
        PREP["tag_vocabulary.py\ DanbooruDataPreprocessor"]
        ANALYZE["Dataset_Analysis.py"]
    end

    subgraph Model["Model & Loss/Metric"]
        ARCH["model_architecture.py\ SimplifiedTagger,\ VisionTransformerConfig:contentReference[oaicite:12]{index=12}"]
        LOSS["loss_functions.py\ AsymmetricFocalLoss,\ MultiTaskLoss"]
        METRICS["metrics.py\ compute_precision_recall_f1():contentReference[oaicite:13]{index=13}"]
        ADV_METRICS["Evaluation_Metrics.py\ MetricComputer,\ MetricConfig:contentReference[oaicite:14]{index=14}"]
    end

    subgraph Train["Training & Validation"]
        TRAIN["train_direct.py\ setup_orientation_aware_training(),\ train_with_orientation_tracking()"]
        VAL["validation_loop.py\ ValidationRunner"]
        UTILS["training_utils.py\ TrainingState,\ DistributedTrainingHelper:contentReference[oaicite:15]{index=15}"]
    end

    subgraph Monitor["Monitoring & Logging"]
        MON["Monitor_log.py\ MonitorConfig,\ TrainingMonitor,\ AlertSystem:contentReference[oaicite:16]{index=16}"]
    end

    subgraph Inference["Inference & Export"]
        INFER["Inference_Engine.py\ InferenceConfig,\ ImagePreprocessor,\ ModelWrapper:contentReference[oaicite:17]{index=17}"]
        ONNX["ONNX_Export.py\ ONNXExporter:contentReference[oaicite:18]{index=18}"]
    end

    %% Edges
    CFG --> LOADER
    CONFIGS --> LOADER
    IGNORES --> VOCAB
    LOADER --> ORIENT --> METADATA
    METADATA --> PREP
    PREP --> VOCAB
    VOCAB --> ARCH
    ARCH --> TRAIN
    LOSS --> TRAIN
    METRICS --> VAL
    ADV_METRICS --> VAL
    UTILS --> TRAIN
    UTILS --> VAL
    TRAIN --> VAL
    MON --- TRAIN
    MON --- VAL
    MON --- INFER
    TRAIN --> INFER
    VAL --> INFER
    INFER --> ONNX
