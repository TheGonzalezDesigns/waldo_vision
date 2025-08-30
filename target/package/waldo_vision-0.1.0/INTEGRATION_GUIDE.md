# Integration Guide: Using `waldo_vision` as a Corpus Filter

**To the Corpus Gemini Agent:**

This guide details how to integrate the `waldo_vision` crate as a high-performance, intelligent pre-filter for the `corpus-vision` capability. This will replace the existing simple heatmap logic with a sophisticated, multi-layered analysis engine.

The goal is to use `waldo_vision` to make a "go/no-go" decision on whether a raw camera frame is significant enough to be sent to the expensive Gemini Analysis API.

---

## 1. Dependency Installation

First, you will need to add the `waldo_vision` crate to the `Cargo.toml` of the `corpus-vision` project.

Since this crate is not yet published on `crates.io`, you can add it directly from its GitHub repository:

```toml
# In corpus-vision/Cargo.toml

[dependencies]
waldo_vision = { git = "https://github.com/TheGonzalezDesigns/waldo_vision.git", branch = "main" }

# ... other dependencies
```

---

## 2. High-Level Architecture

The `waldo_vision` crate exposes a top-level struct called `VisionPipeline`. This is the only object you will need to interact with. It encapsulates the entire analysis engine.

The integration workflow is as follows:
1.  **Initialize**: When the `corpus-vision` service starts, create and store one instance of `VisionPipeline`.
2.  **Configure**: The `VisionPipeline` is configured with a `PipelineConfig` struct, allowing you to tune its sensitivity.
3.  **Process**: For each frame received from the camera, pass the raw image buffer to the pipeline.
4.  **Decide**: Use the simple boolean output to decide whether to proceed with the full Gemini analysis.

---

## 3. Code Implementation

Here is how you can modify the main processing loop within the `corpus-vision` service.

### Step 3.1: Initialization

You will need to import and initialize the pipeline. This should be done once when your service starts.

```rust
// In the main file of corpus-vision, likely where you manage the camera feed.

use waldo_vision::pipeline::{VisionPipeline, PipelineConfig};

// Create a configuration for the pipeline.
// These values should be loaded from a config file in a real application.
let config = PipelineConfig {
    image_width: 1920, // The width of the frames from the camera
    image_height: 1080, // The height of the frames from the camera
    chunk_width: 10,   // A 10x10 chunk size is a good starting point
    chunk_height: 10,
    // A moment must last for at least 15 frames (~0.5s at 30fps) to be considered significant.
    min_moment_age_for_significance: 15,
    // An anomaly must have a statistical significance score of at least 3.5 to be considered.
    significance_threshold: 3.5,
};

// Create and store the pipeline instance.
let mut vision_pipeline = VisionPipeline::new(config);
```

### Step 3.2: Replacing the Old Filter Logic

Locate the part of your code that processes a new frame from the camera. You will be replacing the old `calculate_heatmap` and its associated logic with a single call to our pipeline.

**Old Logic (Conceptual):**

```rust
// This is the code that should be replaced.
fn process_frame(frame_buffer: &[u8]) {
    let heatmap = image_processing::calculate_heatmap(frame_buffer);
    let motion_detected = image_processing::analyze_heatmap(heatmap);

    if motion_detected {
        // Call Gemini API...
    }
}
```

**New, Integrated Logic:**

```rust
// This is the new, much simpler and more powerful logic.
fn process_frame(frame_buffer: &[u8], pipeline: &mut VisionPipeline) {
    // The core of the integration: one simple, fast function call.
    let is_significant = pipeline.significant_mention_detected(frame_buffer);

    if is_significant {
        // The waldo_vision engine has determined this frame contains a
        // statistically significant and persistent event.
        // Now is the time to send the frame to the expensive Gemini API for analysis.
        println!("Significant mention detected! Sending to Gemini...");
        // call_gemini_api(frame_buffer);
    } else {
        // The frame contained no significant events. Do nothing.
        // This saves immense computational cost.
        println!("No significant mention detected.");
    }
}
```

---

## 4. Tuning and Advanced Usage

The power of this integration lies in its tunability and the rich data it makes available.

### Tuning `PipelineConfig`

*   `chunk_width` / `chunk_height`: Smaller values (e.g., 5) are more granular and can detect smaller objects but are more computationally expensive. Larger values (e.g., 20) are faster but might miss very small movements. **A 10x10 grid is a balanced starting point.**
*   `min_moment_age_for_significance`: This is your **persistence filter**. Increase this value to make the system less sensitive to fleeting movements (like a bird flying past the window). Decrease it to be more sensitive to very short events.
*   `significance_threshold`: This is your **statistical sensitivity filter**. It represents the Z-score (standard deviations from the mean) an event must have to be noticed. A value of `3.0` is a good baseline. Increase it to `4.0` or `5.0` to make the system only report on extremely unusual events.

### Advanced: Using the Full `Report`

For more complex scenarios, you can use the `generate_report()` method instead of the simple boolean check. This gives you access to the full data of the detected `Moment`s.

```rust
use waldo_vision::pipeline::Report;

let report = pipeline.generate_report(frame_buffer);

match report {
    Report::NoSignificantMention => {
        // ...
    },
    Report::SignificantMention(mention_data) => {
        // You now have access to the full data of the events.
        for moment in mention_data.new_significant_moments {
            println!("New significant moment started with ID: {}", moment.id);
            // You could, for example, use the data in moment.blob_history
            // to crop the image to the object before sending it to Gemini.
        }
    }
}
```

This advanced usage allows for sophisticated downstream processing, such as cropping the image to the area of interest or feeding the motion path data into another system.

---

This concludes the integration guide. By following these steps, you will successfully replace the existing filter with the `waldo_vision` engine, dramatically increasing the intelligence, performance, and tunability of the `corpus-vision` capability.