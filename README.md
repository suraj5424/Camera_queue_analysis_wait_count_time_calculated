# 📹 Queue Monitoring using YOLOv8 (Simulated Multi-Camera Setup)

This project demonstrates how to monitor queues and count people using **YOLOv8** (from Ultralytics) with a single video, by simulating a multi-camera setup.

> 📽️ **Video Credit:** [March Networks on YouTube](https://youtu.be/5rkwqp6nnr4?si=jdkWhgZ9adCqKLTk)
> 🎯 **Goal:** Track and count people in two queue zones with real-time visualization.

---

## 🚀 Features

* People detection and tracking using **YOLOv8m**
* Real-time **zone-based** counting
* Estimation of **waiting time** for each person
* Simulated **2-camera view** from a single video
* Annotated video output with bounding boxes, labels, and statistics

---

## 🧩 Requirements

Install the necessary packages:

```bash
pip install ultralytics opencv-python imageio matplotlib
```

---

## 📁 Project Structure

```bash
├── main.py                      # Core processing and annotation logic
├── input/
│   └── que.mp4                  # Input video
├── output/
│   └── step4_fully_customizable_output.mp4
├── display.py                   # Code to visualize middle frames
└── README.md
```

---

## 🎥 Video Setup & Methodology

* The original video shows a **top-down camera** above a cashier counter.
* Since only one video is available, it is **logically split** into two zones:

  * **Camera 1 Zone (Left):** Where people enter
  * **Camera 2 Zone (Right):** Where people reach the cashier
* Using **YOLOv8 tracking IDs**, each person is tracked **across both zones** to avoid double-counting.

---

## 🧠 How It Works

1. **Load Video**
2. **Split frame** into two logical zones
3. **Track people** using YOLOv8 with object ID persistence
4. **Count** individuals in each zone
5. **Estimate wait time** in queue
6. **Overlay statistics and bounding boxes**
7. **Save annotated video**

---

## 🖼️ Sample Output (Middle 300 Frames)

```python
from IPython.display import HTML
HTML(display_middle_300("output/step4_fully_customizable_output.mp4").to_html5_video())
```

---

## 🧪 Core Functions

* `get_video_details(path)`: Shows video metadata
* `display_middle_300(path)`: Preview video using 300 central frames
* `is_inside_area()`: Determines if person is in cashier/queue area
* `generate_color(id)`: Assigns consistent color per person ID
* `boxes_intersect()`: Checks bounding box overlap with zone

---

## 🔍 Model Used

* **Model:** `yolov8m.pt`
* **Class:** `0` (person)
* **Confidence Threshold:** Auto-handled by model default
* **Tracking:** Enabled (`model.track()`)

---
### 🧠 Core Idea: **Wait Time Counting / Analysis**

The **wait time analysis** in this project focuses on **measuring how long each person spends in the queue** — from the moment they enter a defined “queue area” until they reach the cashier.

---

### ✅ **How It Works**

1. **Unique ID Tracking**
   Each person is assigned a **persistent ID** by the YOLOv8 tracker.

2. **First Seen Frame**
   The system records the **frame number** when each person’s ID is first detected in the queue area:

   ```python
   if track_id not in id_first_seen:
       id_first_seen[track_id] = frame_number
   ```

3. **Time Calculation**
   For every frame after, the **time waited** is calculated as:

   ```python
   time_waited = (current_frame - first_seen_frame) / fps
   ```

4. **Queue Zone Filtering**
   Time is **only counted** if the person is:

   * In the **queue area** (not random walking)
   * Not yet at the **cashier area**

5. **Real-Time Overlay**
   On the video, the system displays:

   * `ID`, `Confidence`, and `Wait Time`
   * Highlighted zones for Queue and Cashier

---

### 📌 Purpose of Wait Time Analysis

* ✅ **Customer Flow Insight** – Understand how long customers are waiting
* ✅ **Queue Management** – Detect bottlenecks in real-time
* ✅ **Performance Monitoring** – Evaluate service efficiency at counters
* ✅ **No Duplication** – IDs ensure people are counted only once

---

### 🔄 Example Label

A person in the queue might get a label like:

```
ID 12 | 87.3% | Queue | 9.5s
```

This shows:

* Their unique ID
* Detection confidence
* Current zone: Queue
* Total time spent in queue so far

---

## 🧑‍💻 Example Snippet (Main Loop)

```python
results = model.track(frame, persist=True, classes=[0], verbose=False)
for box in results[0].boxes:
    if box.id is None:
        continue
    # Extract tracking ID, bounding box, and confidence
```

---

## 📦 Output

* The final annotated video is saved at:

```
/kaggle/working/step4_fully_customizable_output.mp4
```

It includes:

* Bounding boxes with ID and confidence
* Zone-based person counts
* Wait time for each tracked person
* Visualized cashier and queue areas
---


## 📄 License

This project is licensed under the terms of the [LICENSE.txt](LICENSE.txt). See the file for details.
