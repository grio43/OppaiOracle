-- Dataset Grading Database Schema
-- Stores FP/FN analysis for manual review prioritization

-- Core grading table - one row per image
CREATE TABLE IF NOT EXISTS image_grades (
    id INTEGER PRIMARY KEY,
    image_path TEXT UNIQUE NOT NULL,
    json_path TEXT NOT NULL,
    fp_count INTEGER DEFAULT 0,      -- False Positives (predicted but not in GT)
    fn_count INTEGER DEFAULT 0,      -- False Negatives (in GT but not predicted)
    total_false INTEGER GENERATED ALWAYS AS (fp_count + fn_count) STORED,
    priority_score REAL DEFAULT 0.0, -- Weighted score for review prioritization
    graded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed BOOLEAN DEFAULT FALSE,
    review_decision TEXT CHECK(review_decision IN ('keep', 'fix_tags', 'delete', 'skip', NULL)),
    reviewed_at TIMESTAMP,
    notes TEXT                       -- Reviewer notes
);

-- Per-tag error details for debugging
CREATE TABLE IF NOT EXISTS tag_errors (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES image_grades(id) ON DELETE CASCADE,
    tag_name TEXT NOT NULL,
    error_type TEXT NOT NULL CHECK(error_type IN ('FP', 'FN')),
    confidence REAL,                 -- Model confidence (sigmoid output)
    tag_frequency INTEGER,           -- How common is this tag in dataset
    is_gender_tag BOOLEAN DEFAULT FALSE,
    is_count_tag BOOLEAN DEFAULT FALSE
);

-- Tag corrections for fix_tags decisions
CREATE TABLE IF NOT EXISTS tag_corrections (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES image_grades(id) ON DELETE CASCADE,
    action TEXT NOT NULL CHECK(action IN ('add', 'remove')),
    tag_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit log for all changes
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY,
    image_id INTEGER REFERENCES image_grades(id),
    action TEXT NOT NULL,
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Grading run metadata
CREATE TABLE IF NOT EXISTS grading_runs (
    id INTEGER PRIMARY KEY,
    model_path TEXT NOT NULL,
    threshold REAL NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    total_images INTEGER,
    images_with_errors INTEGER
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_priority ON image_grades(priority_score DESC) WHERE reviewed = FALSE;
CREATE INDEX IF NOT EXISTS idx_total_false ON image_grades(total_false DESC);
CREATE INDEX IF NOT EXISTS idx_reviewed ON image_grades(reviewed);
CREATE INDEX IF NOT EXISTS idx_review_decision ON image_grades(review_decision);
CREATE INDEX IF NOT EXISTS idx_tag_errors_image ON tag_errors(image_id);
CREATE INDEX IF NOT EXISTS idx_tag_errors_tag ON tag_errors(tag_name);
CREATE INDEX IF NOT EXISTS idx_tag_errors_type ON tag_errors(error_type);
CREATE INDEX IF NOT EXISTS idx_corrections_image ON tag_corrections(image_id);

-- Views for common queries
CREATE VIEW IF NOT EXISTS pending_review AS
SELECT
    id, image_path, json_path,
    fp_count, fn_count, total_false, priority_score
FROM image_grades
WHERE reviewed = FALSE AND total_false > 0
ORDER BY priority_score DESC;

CREATE VIEW IF NOT EXISTS review_stats AS
SELECT
    review_decision,
    COUNT(*) as count,
    AVG(total_false) as avg_errors
FROM image_grades
WHERE reviewed = TRUE
GROUP BY review_decision;
