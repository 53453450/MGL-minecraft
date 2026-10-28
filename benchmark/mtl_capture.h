/*
 * mtl_capture.h — optional GPU trace capture for the offline harness.
 *
 * A two-frame run is far too short to attach Xcode's Metal Debugger by hand,
 * so the harness starts a device-scope capture itself and stops it after one
 * frame.  The resulting .gputrace opens in Xcode (File > Open, or double click)
 * and shows every render pass with its attachments and bound textures.
 *
 * Requires MTL_CAPTURE_ENABLED=1 in the environment; without it the capture
 * manager rejects the destination and the calls return -1.
 */
#ifndef MGL_HARNESS_MTL_CAPTURE_H
#define MGL_HARNESS_MTL_CAPTURE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 0 on success, -1 when the capture could not be started. */
int mgl_harness_gpu_trace_begin(const char *path);
int mgl_harness_gpu_trace_end(void);

#ifdef __cplusplus
}
#endif

#endif /* MGL_HARNESS_MTL_CAPTURE_H */
