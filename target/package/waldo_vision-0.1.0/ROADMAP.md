# Waldo Vision - Feature Roadmap

This document outlines the planned features and architectural enhancements for future versions of the `waldo_vision` engine.

---

## Version 0.2.0: Scene Context & Advanced Signatures

*   **Goal**: Enhance the `Moment` data with rich, contextual information about the environment.
*   **Features**:
    *   **Scene Analysis**: Implement logic to analyze the stable background around a `Moment`'s path to generate a "Scene Signature" (e.g., "indoor_doorway", "outdoor_patio").
    *   **Advanced Blob Signatures**: Enhance the `SmartBlob` signature to include more detailed features beyond the anomaly scores, such as:
        *   **Color Histograms**: A more robust representation of an object's color profile.
        *   **Aspect Ratio**: The width/height ratio of the bounding box to distinguish tall objects (people) from long objects (vehicles, animals).
    *   **API Enhancement**: The `Report` and `Moment` structs will be updated to include this new scene and signature data.

---

## Version 0.3.0: Object Re-Identification (Re-ID) & Long-Term Memory

*   **Goal**: Give the system a long-term memory to recognize previously seen objects.
*   **Features**:
    *   **Signature Database**: Create a simple, in-memory database to store the signatures of completed `Moment`s.
    *   **Re-ID Logic**: When a new `Moment` is completed, its signature will be compared against the database to find potential matches.
    *   **"Known Actor" Filter**: The `VisionPipeline` will be enhanced with a new filter to allow users to suppress notifications for recognized, routine objects (e.g., "ignore the family dog").
    *   **API Enhancement**: The `Moment` struct will be updated with an optional `reidentified_as_id` field.

---

## Version 1.0.0: Full Behavioral Analysis

*   **Goal**: Transition from event detection to true behavioral analysis.
*   **Features**:
    *   **Pattern-of-Life Analysis**: Implement logic to analyze the historical archive of `Moment`s to find patterns (e.g., "Object #42 usually appears in Scene 'doorway' between 8 AM and 9 AM").
    *   **Motion Pattern Recognition**: Develop heuristics to classify the `path` data within a `Moment` into categories like "pacing," "stationary," "linear motion," or "erratic motion."
    *   **Advanced Filtering API**: Expose a powerful filtering API to allow users to request notifications for highly specific behaviors (e.g., "only notify me if an unknown object appears and moves erratically in the 'patio' scene at night").

---
