# Multi-Behavior Labeling - Fixes Applied

## ✅ Critical Issues Fixed

### 1. Table Display - Now Shows Labels
- **Fixed**: `update_table()` method now includes label column
- **Fixed**: Table initialized with 5 columns instead of 4
- **Fixed**: Labels are displayed in the table for each event
- **Location**: `src/behavex/annotation/pavs.py` lines 568-703, 1435-1436

### 2. Behavior Selector GUI Added
- **Added**: `QComboBox` widget for selecting behavior category
- **Added**: `get_available_behaviors()` method to get behaviors from project config or defaults
- **Added**: Behavior selector appears in GUI above the table
- **Added**: `on_behavior_changed()` callback to update current behavior
- **Location**: `src/behavex/annotation/pavs.py` lines 75-88, 381-389, 504-505, 595-609

### 3. build_labels() Now Filters by Behavior
- **Updated**: `build_labels()` now accepts optional `behavior_filter` parameter
- **Updated**: Filters annotations by label column when behavior is specified
- **Updated**: Backward compatible - works with or without label column
- **Updated**: `process_sessions_for_training()` passes behavior filter from config
- **Location**: `src/behavex/project/session.py` lines 257-281
- **Location**: `src/behavex/project/project.py` lines 449-461

### 4. Additional Fixes
- **Fixed**: `clearTable()` now clears `events_labels` list
- **Fixed**: `importCSV()` now handles labels properly
- **Fixed**: Table column count updated from 4 to 5 in `insertBaseRow()`

## 🔧 Implementation Details

### Behavior Selector
- Default behaviors: ["rearing", "grooming", "feeding", "drinking", "exploring", "resting"]
- Can be configured via project config: `annotation.behaviors`
- Falls back to single `annotation.behavior` field for backward compatibility
- Prevents changing behavior while event is open

### Label Filtering
- When `behavior_filter` is specified, only events with matching label are included
- If no label column exists, all events are included (backward compatible)
- Warning message shown if no events found for specified behavior

### CSV Format
- Header: `['index', 'start_frame', 'end_frame', 'duration', 'label']`
- Backward compatible with old 4-column format
- Defaults to "rearing" for old files without labels

## 📋 Remaining Tasks (Optional)

1. **Config Structure Enhancement**: Update default config to support list of behaviors
2. **Prediction Methods**: Update to handle behavior categories in predictions
3. **Documentation**: Update API docs with new multi-behavior features

## ✨ Usage

### Selecting Behavior in GUI
1. Use the "Behavior:" dropdown above the annotation table
2. Select desired behavior before marking events
3. All new events will use selected behavior label

### Training with Behavior Filter
```python
# In config.yaml:
annotation:
  behavior_filter: "rearing"  # Train only on rearing events
  
# Or set programmatically:
project.set_config_value("annotation.behavior_filter", "rearing")
project.process_sessions_for_training()
```

### Building Labels for Specific Behavior
```python
# Filter labels by behavior
session.build_labels(behavior_filter="grooming")

# Or build labels for all behaviors (default)
session.build_labels()
```

