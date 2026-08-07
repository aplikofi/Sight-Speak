# SightSpeak
This project is an Eye-Gaze Virtual Keyboard designed to help individuals with severe motor impairments—such as ALS, locked-in syndrome, muscular dystrophy, or spinal cord injuries—to write and communicate using only their eye movements and a standard webcam.

Here is a breakdown of how the project works and how it facilitates communication:

1. Eye Tracking via Webcam (No Special Hardware Required)
Instead of requiring expensive infrared eye-trackers, this system uses a standard computer webcam. It leverages Google's MediaPipe Face Landmarker to track facial features in real-time. By measuring the relative position of the iris/pupil landmarks (across 10 iris coordinates) relative to the eye corners, it calculates horizontal and vertical gaze ratios.

2. Dwell-to-Select Typing (No Clicks Needed)
For users who cannot press physical keys or click a mouse, the interface operates on a dwell-time selection system:

The user moves their eyes to look at a key on the virtual keyboard.
A visual cursor follows their gaze on the screen.
When they look (hover) at a key for a set amount of time (e.g., 1.2 seconds), a progress bar fills up, and the letter is automatically typed.
The dwell duration is fully adjustable to accommodate different user speeds.
3. Smart Calibration and Head-Pose Compensation
To make eye tracking accurate and practical for daily use:

9-Point Calibration: The system guides the user through looking at a 9-point grid on the screen to calculate a quadratic mapping function from their unique eye movements to screen coordinates.
Head Pose Compensation: It automatically estimates head pitch and yaw to offset minor head movements. This prevents the cursor from drifting if the user adjusts their posture or tilts their head.
Implicit Recalibration: As the user types, the system continuously refines its accuracy by assuming that successful key selections mean the user was looking at the center of that key, adapting to drift dynamically.
4. Smoothing and Jitter Reduction
Eyes are naturally prone to small, rapid involuntary movements (micro-saccades). The project implements a One-Euro Filter (an adaptive smoothing filter) to stabilize the cursor. This keeps the cursor rock-steady when the user is trying to focus on a single key, but allows it to move quickly and responsively when they look at a different part of the screen.
