# Latest pending LiDAR experiment

This branch changes only the threaded frontend's overflow choice: convert the
incoming scan successfully, then drop the oldest pending scan if the queue is
still full under its mutex. The currently registering scan is untouched. IMU
readiness is recomputed against the new queue head. Capacity must be positive.
There is no new user parameter; this is not a mainline or preset change.

Motivation: the NDT-only 4/2-thread CPU ABBA experiment failed in one 2-thread
run, with frontend drift preceding native localization error. Existing FIFO1
keeps the older pending scan and discards incoming scans before conversion.
This motivates a comparison, but does not establish queue policy as the cause.

`test_latest_pending` calls the real ThreadedNode callbacks and checks bounded
replacement, failed-conversion preservation, IMU readiness after replacement,
worker consumption/wakeup and shutdown, and invalid capacity. Queue inspection
uses a joined worker or its mutex. It does not establish trajectory accuracy,
DDS delivery, or real-time performance under contention.

Before adoption, compare normal Mix and a fresh CPU ABBA with identical
frontend/native 4-thread settings, FIFO1 and localizer 9b65a6ba. Keep the old
runtime immutable. Record point-cloud and TF coverage, accuracy, output gaps,
CPU and memory: converting incoming scans before overflow can cost more CPU.
Broaden normal/map-input checks only if the candidate has measurable benefit.
