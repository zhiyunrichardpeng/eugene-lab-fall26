@echo off
echo Starting FFMPEG processing...

echo.
echo "--- [1/1] Processing: v95_test_30_frames ---"
ffmpeg -framerate 22 -start_number 14 -i "C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub\temp_rendered_png_hub\face_closeup\beauty\v95_test_30_frames_1_%%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 23 -preset medium -vf "scale=1024:1024:flags=lanczos" "C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\results\DSAC_V2_m1_vertexMove_Va003\maya_render_output_hub\o59-25-10-8\v95_test_30_frames.mp4"

@echo on
echo.
echo All tasks complete.
