"""Validation utilities for checking consistency between videos, features, labels, and predictions."""
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import cv2


def get_video_frame_count(video_path: Path) -> Optional[int]:
    """Extract frame count from video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Frame count if video can be read, None otherwise
    """
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.exists():
        return None
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    except Exception:
        return None


def validate_features(
    features_length: int,
    video_frame_count: Optional[int] = None,
    tolerance: float = 0.01,
    strict_mode: bool = False
) -> Tuple[bool, List[str]]:
    """Validate features length against video frame count.
    
    Args:
        features_length: Number of rows in features
        video_frame_count: Expected frame count from video (optional)
        tolerance: Allowed relative difference (default: 1%)
        strict_mode: If True, raise errors instead of warnings
        
    Returns:
        Tuple of (is_valid, list_of_messages)
    """
    messages = []
    is_valid = True
    
    if video_frame_count is None:
        return True, messages
    
    diff = abs(features_length - video_frame_count)
    relative_diff = diff / video_frame_count if video_frame_count > 0 else 0
    
    if relative_diff > tolerance:
        msg = (
            f"Features length ({features_length}) does not match video frame count "
            f"({video_frame_count}). Difference: {diff} frames ({relative_diff*100:.2f}%)"
        )
        messages.append(msg)
        is_valid = False
        
        if strict_mode:
            raise ValueError(msg)
        else:
            print(f"Warning: {msg}")
    
    return is_valid, messages


def validate_annotations(
    annotations: pd.DataFrame,
    data_length: int,
    strict_mode: bool = False
) -> Tuple[bool, List[str]]:
    """Validate annotation frame ranges against data length.
    
    Args:
        annotations: DataFrame with start_frame and end_frame columns
        data_length: Expected maximum frame index (features length or video frame count)
        strict_mode: If True, raise errors instead of warnings
        
    Returns:
        Tuple of (is_valid, list_of_messages)
    """
    messages = []
    is_valid = True
    
    if annotations is None or len(annotations) == 0:
        return True, messages
    
    if "start_frame" not in annotations.columns or "end_frame" not in annotations.columns:
        msg = "Annotations missing required columns: start_frame, end_frame"
        messages.append(msg)
        is_valid = False
        if strict_mode:
            raise ValueError(msg)
        else:
            print(f"Warning: {msg}")
        return is_valid, messages
    
    invalid_ranges = []
    for idx, row in annotations.iterrows():
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        
        if start < 0:
            invalid_ranges.append(f"Row {idx}: start_frame ({start}) < 0")
        if end >= data_length:
            invalid_ranges.append(f"Row {idx}: end_frame ({end}) >= data_length ({data_length})")
        if start > end:
            invalid_ranges.append(f"Row {idx}: start_frame ({start}) > end_frame ({end})")
    
    if invalid_ranges:
        msg = f"Found {len(invalid_ranges)} invalid annotation ranges:\n" + "\n".join(invalid_ranges[:5])
        if len(invalid_ranges) > 5:
            msg += f"\n... and {len(invalid_ranges) - 5} more"
        messages.append(msg)
        is_valid = False
        
        if strict_mode:
            raise ValueError(msg)
        else:
            print(f"Warning: {msg}")
    
    return is_valid, messages


def validate_labels(
    labels: np.ndarray,
    features_length: int,
    strict_mode: bool = False
) -> Tuple[bool, List[str]]:
    """Validate labels length matches features length.
    
    Args:
        labels: Labels array
        features_length: Expected length (features.shape[0])
        strict_mode: If True, raise errors instead of warnings
        
    Returns:
        Tuple of (is_valid, list_of_messages)
    """
    messages = []
    is_valid = True
    
    if labels is None:
        msg = "Labels are None"
        messages.append(msg)
        is_valid = False
        if strict_mode:
            raise ValueError(msg)
        return is_valid, messages
    
    labels_length = len(labels) if labels.ndim == 1 else labels.shape[0]
    
    if labels_length != features_length:
        msg = (
            f"Labels length ({labels_length}) does not match features length "
            f"({features_length})"
        )
        messages.append(msg)
        is_valid = False
        
        if strict_mode:
            raise ValueError(msg)
        else:
            print(f"Warning: {msg}")
    
    return is_valid, messages


def validate_predictions(
    predictions: np.ndarray,
    source_length: int,
    strict_mode: bool = False
) -> Tuple[bool, List[str]]:
    """Validate predictions length matches source data length.
    
    Args:
        predictions: Predictions array (1D for binary, 2D for multiclass)
        source_length: Expected length (features or windows length)
        strict_mode: If True, raise errors instead of warnings
        
    Returns:
        Tuple of (is_valid, list_of_messages)
    """
    messages = []
    is_valid = True
    
    if predictions is None:
        msg = "Predictions are None"
        messages.append(msg)
        is_valid = False
        if strict_mode:
            raise ValueError(msg)
        return is_valid, messages
    
    pred_length = predictions.shape[0]
    
    if pred_length != source_length:
        msg = (
            f"Predictions length ({pred_length}) does not match source data length "
            f"({source_length})"
        )
        messages.append(msg)
        is_valid = False
        
        if strict_mode:
            raise ValueError(msg)
        else:
            print(f"Warning: {msg}")
    
    return is_valid, messages


def validate_session_consistency(
    session,
    video_frame_count: Optional[int] = None,
    strict_mode: bool = False,
    enable_checks: bool = True
) -> Dict[str, any]:
    """Main validation function for a session.
    
    Args:
        session: Session object to validate
        video_frame_count: Optional video frame count (will be extracted if not provided)
        strict_mode: If True, raise errors instead of warnings
        enable_checks: If False, skip all checks
        
    Returns:
        Dictionary with validation results:
        {
            'is_valid': bool,
            'messages': List[str],
            'warnings': List[str],
            'errors': List[str],
            'checks': {
                'video_features': bool,
                'features_annotations': bool,
                'features_labels': bool,
                'features_predictions': bool
            }
        }
    """
    result = {
        'is_valid': True,
        'messages': [],
        'warnings': [],
        'errors': [],
        'checks': {
            'video_features': None,
            'features_annotations': None,
            'features_labels': None,
            'features_predictions': None
        }
    }
    
    if not enable_checks:
        return result
    
    # Get video frame count if not provided
    if video_frame_count is None and session.annotation_view:
        try:
            video_path = session.path_to_view(session.annotation_view)
            video_frame_count = get_video_frame_count(video_path)
        except (KeyError, ValueError):
            pass
    
    # Check 1: Video ↔ Features
    if session.has_features():
        try:
            features_df = pd.read_csv(session.features_path())
            features_length = len(features_df)
            
            valid, msgs = validate_features(
                features_length, video_frame_count, strict_mode=strict_mode
            )
            result['checks']['video_features'] = valid
            result['messages'].extend(msgs)
            if not valid:
                result['is_valid'] = False
                if strict_mode:
                    result['errors'].extend(msgs)
                else:
                    result['warnings'].extend(msgs)
        except Exception as e:
            msg = f"Error validating features: {e}"
            result['messages'].append(msg)
            result['warnings'].append(msg)
    
    # Check 2: Features ↔ Annotations
    if session.has_annotation():
        try:
            session.load_annotations()
            annotations = session.annotations
            
            # Get data length from features if available, otherwise video
            data_length = None
            if session.has_features():
                try:
                    features_df = pd.read_csv(session.features_path())
                    data_length = len(features_df)
                except Exception:
                    pass
            
            if data_length is None and video_frame_count is not None:
                data_length = video_frame_count
            
            if data_length is not None:
                valid, msgs = validate_annotations(
                    annotations, data_length, strict_mode=strict_mode
                )
                result['checks']['features_annotations'] = valid
                result['messages'].extend(msgs)
                if not valid:
                    result['is_valid'] = False
                    if strict_mode:
                        result['errors'].extend(msgs)
                    else:
                        result['warnings'].extend(msgs)
        except Exception as e:
            msg = f"Error validating annotations: {e}"
            result['messages'].append(msg)
            result['warnings'].append(msg)
    
    # Check 3: Features ↔ Labels
    if session.labels is not None and session.features is not None:
        try:
            features_length = session.features.shape[0]
            valid, msgs = validate_labels(
                session.labels, features_length, strict_mode=strict_mode
            )
            result['checks']['features_labels'] = valid
            result['messages'].extend(msgs)
            if not valid:
                result['is_valid'] = False
                if strict_mode:
                    result['errors'].extend(msgs)
                else:
                    result['warnings'].extend(msgs)
        except Exception as e:
            msg = f"Error validating labels: {e}"
            result['messages'].append(msg)
            result['warnings'].append(msg)
    
    # Check 4: Features ↔ Predictions
    if session.has_predictions():
        try:
            predictions = np.load(session.predictions_path())
            source_length = None
            
            if session.features is not None:
                source_length = session.features.shape[0]
            elif session.has_features():
                try:
                    features_df = pd.read_csv(session.features_path())
                    source_length = len(features_df)
                except Exception:
                    pass
            
            if source_length is not None:
                valid, msgs = validate_predictions(
                    predictions, source_length, strict_mode=strict_mode
                )
                result['checks']['features_predictions'] = valid
                result['messages'].extend(msgs)
                if not valid:
                    result['is_valid'] = False
                    if strict_mode:
                        result['errors'].extend(msgs)
                    else:
                        result['warnings'].extend(msgs)
        except Exception as e:
            msg = f"Error validating predictions: {e}"
            result['messages'].append(msg)
            result['warnings'].append(msg)
    
    return result

